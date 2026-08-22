# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded autonomous repair controller facade.

``AutonomousRepairController@1`` selects among deterministic, template-
constrained, and model-assisted tiers, then delegates admission and
execution to the existing ``autonomous_repair`` engine and materializer.
It is not a second repair engine, mutation path, merge authority, or
completion authority.

Model-assisted repair requires exact files/symbols, a bounded patch
envelope, sufficient context, an isolated worktree, predetermined
tests/proofs, and protected authority paths.  Repeated identical
failures reuse diagnosis and back off without another model call.

Autonomous merge is a conjunction of every stated low-risk condition.
R3 remains proposal-only.  Repair receipts never independently
authorize merge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..autonomous_repair.engine import AutonomousRepairEngine
from ..autonomous_repair.materialize import (
    AdmittedSourceEditError,
    AdmittedSourceEditOperator,
    AutonomousRepairMaterializer,
)
from ..proof.formal_verification_contracts import canonical_json, content_identity
from .contracts import (
    AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
    MAX_CANONICAL_RECORD_BYTES,
    MAX_IDENTIFIER_BYTES,
    MAX_MAPPING_ITEMS,
    MAX_SEQUENCE_ITEMS,
    AutonomousRepairPlan,
    AutonomousRepairReceipt,
    AutonomyEnvelope,
    AutonomyLevel,
    AutonomyPolicy,
    RepairTier,
    RiskClass,
    TerminalStatus,
)
from .receding_horizon import PlanSuffixInvalidationReceipt

AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE: Final[str] = "AutonomousRepairController@1"
REPAIR_CONTROLLER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-result@1"
)
REPAIR_CONTROLLER_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-snapshot@1"
)
REPAIR_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-request@1"
)
MAX_REPAIR_CONTROLLER_SNAPSHOT_BYTES: Final[int] = 4 * MAX_CANONICAL_RECORD_BYTES
DEFAULT_IDENTICAL_FAILURE_BACKOFF_MS: Final[int] = 10
MAX_IDENTICAL_FAILURE_BACKOFF_MS: Final[int] = 40
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 4
SELF_EDIT_PATH: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
)

PROTECTED_AUTHORITY_PREFIXES: Final[tuple[str, ...]] = (
    SELF_EDIT_PATH,
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair/",
    "ipfs_accelerate_py/agent_supervisor/validation/",
    "ipfs_accelerate_py/agent_supervisor/verification/",
    "trusted_keys/",
    "secrets/",
    "credentials/",
)
PROTECTED_AUTHORITY_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        "policy.json",
        "policy.yaml",
        "policy.yml",
        "validator_policy.json",
        "validator-policy-key",
        "validator_policy_key",
        "authorized_keys",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
    }
)
PROTECTED_AUTHORITY_SEGMENTS: Final[frozenset[str]] = frozenset(
    {
        "trusted_keys",
        "private_keys",
        "signing_keys",
        "secrets",
        "credentials",
    }
)
PROTECTED_AUTHORITY_SUFFIXES: Final[tuple[str, ...]] = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
)
LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "policy_autonomous_merge_enabled",
    "risk_at_most_r2",
    "reversible",
    "not_security_sensitive",
    "not_protocol_sensitive",
    "not_irreversible_external",
    "not_legal_or_financial",
    "tier_not_model_assisted",
    "autonomy_level_allows_execute_reversible",
    "required_tests_present",
    "required_proofs_present",
    "validation_receipts_current",
    "rollback_plan_bound",
    "no_protected_authority_mutation",
    "paths_within_envelope",
    "repair_succeeded",
    "isolated_worktree_when_required",
    "predetermined_checks_bound",
)
_TIER_RANK: Final[Mapping[RepairTier, int]] = MappingProxyType(
    {
        RepairTier.DETERMINISTIC: 0,
        RepairTier.TEMPLATE_CONSTRAINED: 1,
        RepairTier.MODEL_ASSISTED_BOUNDED: 2,
    }
)
_FORBIDDEN_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "chain_of_thought",
        "cookie",
        "credential",
        "decoded_source",
        "executable_code",
        "hidden_reasoning",
        "model_transcript",
        "password",
        "private_key",
        "prompt",
        "raw_prompt",
        "refresh_token",
        "secret",
        "shell_command",
        "source_body",
        "transcript",
    }
)


class RepairControllerError(ValueError):
    """Raised when repair-controller inputs themselves are malformed."""


class RepairControllerDisposition(str, Enum):
    """Closed outcome of one facade step."""

    ADMITTED = "admitted"
    EXECUTED = "executed"
    REJECTED = "rejected"
    IDENTICAL_FAILURE_BACKOFF = "identical_failure_backoff"
    ROLLBACK = "rollback"


class RepairMergeDisposition(str, Enum):
    """Closed merge recommendation.  Never merge authority."""

    AUTONOMOUS_MERGE_CANDIDATE = "autonomous_merge_candidate"
    PROPOSAL_ONLY = "proposal_only"
    REJECTED = "rejected"
    NOT_APPLICABLE = "not_applicable"


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise RepairControllerError(f"{name} must be a compact identifier")
    if required and not result:
        raise RepairControllerError(f"{name} is required")
    if (
        len(result.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in result)
        or "\x00" in result
        or any(ord(char) < 32 for char in result)
    ):
        if result:
            raise RepairControllerError(f"{name} must be a compact bounded identifier")
    return result


def _identifiers(
    value: Any,
    name: str,
    *,
    required: bool = False,
    preserve_order: bool = True,
    maximum: int = MAX_SEQUENCE_ITEMS,
) -> tuple[str, ...]:
    if value is None:
        raw: Sequence[Any] = ()
    elif isinstance(value, str):
        raw = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise RepairControllerError(f"{name} must be a sequence of identifiers")
    if len(raw) > maximum:
        raise RepairControllerError(f"{name} contains too many items")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        identifier = _identifier(item, name)
        if identifier not in seen:
            seen.add(identifier)
            normalized.append(identifier)
    if required and not normalized:
        raise RepairControllerError(f"{name} must not be empty")
    return tuple(normalized if preserve_order else sorted(normalized))


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise RepairControllerError(f"{name} must be one of: {allowed}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RepairControllerError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RepairControllerError(f"{name} must be an integer of at least {minimum}")
    return value


def _reject_forbidden_keys(payload: Mapping[str, Any], name: str) -> None:
    for key in payload:
        if not isinstance(key, str):
            raise RepairControllerError(f"{name} keys must be strings")
        normalized = key.strip().lower().replace("-", "_")
        if any(
            normalized == marker or normalized.endswith("_" + marker)
            for marker in _FORBIDDEN_FIELD_MARKERS
        ):
            raise RepairControllerError(
                f"{name} contains forbidden private or executable data"
            )


def _posix_path(value: Any, name: str) -> str:
    text = _identifier(value, name)
    parsed = PurePosixPath(text.replace("\\", "/"))
    if parsed.is_absolute() or ".." in parsed.parts or text in {".", ""}:
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    return parsed.as_posix()


def _paths(
    value: Any,
    name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    raw = _identifiers(value, name, required=required, preserve_order=True)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        path = _posix_path(item, name)
        if path not in seen:
            seen.add(path)
            normalized.append(path)
    return tuple(normalized)


def path_within_allowed(path: str, allowed: Sequence[str]) -> bool:
    """Return True when *path* equals or descends from an allowed prefix."""

    candidate = path.replace("\\", "/").lstrip("./")
    for prefix in allowed:
        bound = prefix.replace("\\", "/").rstrip("/")
        if candidate == bound or candidate.startswith(bound + "/"):
            return True
    return False


def protected_authority_reason(path: str) -> str | None:
    """Return a closed rejection code when *path* is a protected authority surface."""

    try:
        normalized = _posix_path(path, "path")
    except RepairControllerError:
        return "scope_escape"
    lowered = normalized.lower()
    if normalized == SELF_EDIT_PATH or lowered == SELF_EDIT_PATH.lower():
        return "self_edit"
    for prefix in PROTECTED_AUTHORITY_PREFIXES:
        bound = prefix.rstrip("/")
        if normalized == bound or normalized.startswith(bound + "/"):
            if bound == SELF_EDIT_PATH:
                return "self_edit"
            if bound.endswith("validation") or "/validation/" in f"{bound}/":
                return "validator_policy_key"
            if bound in {"trusted_keys", "secrets", "credentials"}:
                return "validator_policy_key"
            return "protected_authority"
    name = PurePosixPath(lowered).name
    if name in PROTECTED_AUTHORITY_BASENAMES:
        return "validator_policy_key"
    if any(lowered.endswith(suffix) for suffix in PROTECTED_AUTHORITY_SUFFIXES):
        return "validator_policy_key"
    if any(segment in PROTECTED_AUTHORITY_SEGMENTS for segment in PurePosixPath(lowered).parts):
        return "validator_policy_key"
    if "validator-policy-key" in lowered or "validator_policy_key" in lowered:
        return "validator_policy_key"
    return None


def _scope_rejection(paths: Sequence[str], allowed: Sequence[str]) -> str | None:
    for path in paths:
        reason = protected_authority_reason(path)
        if reason is not None:
            return reason
        if allowed and not path_within_allowed(path, allowed):
            return "scope_escape"
    return None


@dataclass(frozen=True)
class RepairRequest:
    """One bounded repair request.  Never a prompt, transcript, or patch body."""

    envelope: AutonomyEnvelope
    policy: AutonomyPolicy
    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    preferred_tier: RepairTier = RepairTier.DETERMINISTIC
    worktree_id: str = ""
    context_reference_ids: tuple[str, ...] = ()
    required_test_ids: tuple[str, ...] = ()
    required_proof_ids: tuple[str, ...] = ()
    rollback_plan_id: str = "rollback:repair-default"
    forbidden_symbols: tuple[str, ...] = ()
    max_changed_files: int = 8
    max_changed_lines: int = 400
    work_items: tuple[Any, ...] = ()
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""
    source_edit_operator: Mapping[str, Any] | None = None
    suffix_receipt: PlanSuffixInvalidationReceipt | None = None
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    allow_code_edit_materialize: bool = False
    edit_plan: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, AutonomyEnvelope):
            raise RepairControllerError("repair request requires an AutonomyEnvelope")
        if not isinstance(self.policy, AutonomyPolicy):
            raise RepairControllerError("repair request requires an AutonomyPolicy")
        object.__setattr__(
            self,
            "preferred_tier",
            _enum(self.preferred_tier, RepairTier, "preferred_tier"),
        )
        object.__setattr__(
            self, "predicted_files", _paths(self.predicted_files, "predicted_files", required=True)
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _identifiers(self.predicted_symbols, "predicted_symbols", required=True),
        )
        object.__setattr__(
            self,
            "worktree_id",
            _identifier(self.worktree_id, "worktree_id", required=False),
        )
        for name in (
            "context_reference_ids",
            "required_test_ids",
            "required_proof_ids",
            "forbidden_symbols",
            "validation_receipt_ids",
            "proof_receipt_ids",
            "adversarial_assurance_receipt_ids",
        ):
            object.__setattr__(
                self,
                name,
                _identifiers(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "rollback_plan_id",
            _identifier(self.rollback_plan_id, "rollback_plan_id"),
        )
        object.__setattr__(
            self,
            "max_changed_files",
            _int(self.max_changed_files, "max_changed_files", minimum=1),
        )
        object.__setattr__(
            self,
            "max_changed_lines",
            _int(self.max_changed_lines, "max_changed_lines", minimum=1),
        )
        if self.work_items is None:
            object.__setattr__(self, "work_items", ())
        elif isinstance(self.work_items, (str, bytes, bytearray)):
            raise RepairControllerError("work_items must be a sequence of work items")
        elif not isinstance(self.work_items, Sequence):
            raise RepairControllerError("work_items must be a sequence of work items")
        elif len(self.work_items) > MAX_SEQUENCE_ITEMS:
            raise RepairControllerError("work_items contains too many items")
        else:
            object.__setattr__(self, "work_items", tuple(self.work_items))
        object.__setattr__(
            self,
            "failure_signature",
            _identifier(self.failure_signature, "failure_signature", required=False),
        )
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _identifier(self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False),
        )
        object.__setattr__(
            self,
            "allow_code_edit_materialize",
            _bool(self.allow_code_edit_materialize, "allow_code_edit_materialize"),
        )
        if self.source_edit_operator is not None:
            if not isinstance(self.source_edit_operator, Mapping):
                raise RepairControllerError("source_edit_operator must be a mapping")
            if len(self.source_edit_operator) > MAX_MAPPING_ITEMS:
                raise RepairControllerError("source_edit_operator contains too many fields")
            _reject_forbidden_keys(self.source_edit_operator, "source_edit_operator")
            object.__setattr__(
                self, "source_edit_operator", MappingProxyType(dict(self.source_edit_operator))
            )
        if self.edit_plan is not None:
            if not isinstance(self.edit_plan, Mapping):
                raise RepairControllerError("edit_plan must be a mapping")
            _reject_forbidden_keys(self.edit_plan, "edit_plan")
            object.__setattr__(self, "edit_plan", MappingProxyType(dict(self.edit_plan)))
        if self.suffix_receipt is not None and not isinstance(
            self.suffix_receipt, PlanSuffixInvalidationReceipt
        ):
            raise RepairControllerError("suffix_receipt must be a PlanSuffixInvalidationReceipt")
        colliding = set(self.predicted_symbols).intersection(self.forbidden_symbols)
        if colliding:
            raise RepairControllerError("predicted symbols collide with forbidden symbols")

    @property
    def effective_test_ids(self) -> tuple[str, ...]:
        return self.required_test_ids or self.envelope.required_test_ids

    @property
    def effective_proof_ids(self) -> tuple[str, ...]:
        return self.required_proof_ids or self.envelope.required_proof_ids

    @property
    def effective_context_ids(self) -> tuple[str, ...]:
        return self.context_reference_ids or (self.envelope.envelope_id,)


@dataclass(frozen=True)
class RepairControllerResult:
    """Facade result.  Evidence only; never merge or completion authority."""

    disposition: RepairControllerDisposition
    merge_disposition: RepairMergeDisposition
    reason_codes: tuple[str, ...]
    selected_tier: RepairTier | None = None
    plan: AutonomousRepairPlan | None = None
    receipt: AutonomousRepairReceipt | None = None
    engine_report: Mapping[str, Any] | None = None
    materialize_report: Mapping[str, Any] | None = None
    low_risk_conditions: Mapping[str, bool] = field(default_factory=dict)
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    model_call_count: int = 0
    engine_invoked: bool = False
    mutation_applied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairControllerDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "merge_disposition",
            _enum(self.merge_disposition, RepairMergeDisposition, "merge_disposition"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", preserve_order=True),
        )
        if self.selected_tier is not None:
            object.__setattr__(
                self, "selected_tier", _enum(self.selected_tier, RepairTier, "selected_tier")
            )
        if self.plan is not None and not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan")
        if self.receipt is not None and not isinstance(self.receipt, AutonomousRepairReceipt):
            raise RepairControllerError("receipt must be an AutonomousRepairReceipt")
        if self.receipt is not None and self.receipt.authorizes_merge:
            raise RepairControllerError("repair receipts cannot independently authorize merge")
        object.__setattr__(
            self, "diagnostic_reused", _bool(self.diagnostic_reused, "diagnostic_reused")
        )
        object.__setattr__(
            self,
            "backoff_milliseconds",
            _int(self.backoff_milliseconds, "backoff_milliseconds"),
        )
        object.__setattr__(
            self, "model_call_count", _int(self.model_call_count, "model_call_count")
        )
        object.__setattr__(self, "engine_invoked", _bool(self.engine_invoked, "engine_invoked"))
        object.__setattr__(
            self, "mutation_applied", _bool(self.mutation_applied, "mutation_applied")
        )
        conditions = self.low_risk_conditions or {}
        if not isinstance(conditions, Mapping):
            raise RepairControllerError("low_risk_conditions must be a mapping")
        normalized = {
            _identifier(key, "low_risk_conditions"): _bool(value, "low_risk_conditions")
            for key, value in conditions.items()
        }
        object.__setattr__(self, "low_risk_conditions", MappingProxyType(normalized))
        if self.engine_report is not None:
            if not isinstance(self.engine_report, Mapping):
                raise RepairControllerError("engine_report must be a mapping")
            object.__setattr__(self, "engine_report", MappingProxyType(dict(self.engine_report)))
        if self.materialize_report is not None:
            if not isinstance(self.materialize_report, Mapping):
                raise RepairControllerError("materialize_report must be a mapping")
            object.__setattr__(
                self, "materialize_report", MappingProxyType(dict(self.materialize_report))
            )

    @property
    def authorizes_merge(self) -> bool:
        return False

    @property
    def authorizes_effect(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_CONTROLLER_RESULT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "disposition": self.disposition.value,
            "merge_disposition": self.merge_disposition.value,
            "reason_codes": list(self.reason_codes),
            "selected_tier": None if self.selected_tier is None else self.selected_tier.value,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "engine_report": None if self.engine_report is None else dict(self.engine_report),
            "materialize_report": (
                None if self.materialize_report is None else dict(self.materialize_report)
            ),
            "low_risk_conditions": dict(self.low_risk_conditions),
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "model_call_count": self.model_call_count,
            "engine_invoked": self.engine_invoked,
            "mutation_applied": self.mutation_applied,
            "authorizes_merge": False,
            "authorizes_effect": False,
            "completion_authoritative": False,
        }


def evaluate_low_risk_merge_conjunction(
    *,
    request: RepairRequest,
    plan: AutonomousRepairPlan | None,
    receipt: AutonomousRepairReceipt | None,
    changed_paths: Sequence[str] = (),
) -> Mapping[str, bool]:
    """Evaluate every stated low-risk merge condition.  Conjunction, not scoring."""

    envelope = request.envelope
    risk = envelope.risk_assessment
    policy = request.policy
    paths = tuple(changed_paths) or (plan.predicted_files if plan is not None else ())
    protected = any(protected_authority_reason(path) is not None for path in paths)
    escaped = any(not path_within_allowed(path, envelope.allowed_paths) for path in paths)
    tests = request.effective_test_ids
    proofs = request.effective_proof_ids
    succeeded = (
        receipt is not None
        and receipt.terminal_status is TerminalStatus.SUCCEEDED
        and bool(receipt.validation_receipt_ids)
    )
    worktree_ok = (
        plan is None
        or plan.repair_tier is not RepairTier.MODEL_ASSISTED_BOUNDED
        or bool(plan.worktree_id)
    )
    checks_bound = bool(tests) and (not proofs or (receipt is not None and receipt.proof_receipt_ids))
    conditions = {
        "policy_autonomous_merge_enabled": policy.autonomous_merge_enabled,
        "risk_at_most_r2": risk.risk_class.rank <= RiskClass.R2_REVERSIBLE_LOCAL.rank,
        "reversible": bool(risk.reversible and envelope.reversible),
        "not_security_sensitive": not risk.security_sensitive,
        "not_protocol_sensitive": not risk.protocol_sensitive,
        "not_irreversible_external": not risk.irreversible_external_effect,
        "not_legal_or_financial": not risk.legal_or_financial_effect,
        "tier_not_model_assisted": (
            plan is not None and plan.repair_tier is not RepairTier.MODEL_ASSISTED_BOUNDED
        ),
        "autonomy_level_allows_execute_reversible": (
            envelope.autonomy_level.rank >= AutonomyLevel.EXECUTE_REVERSIBLE.rank
            and policy.allows(AutonomyLevel.EXECUTE_REVERSIBLE, risk.risk_class)
        ),
        "required_tests_present": bool(tests)
        and (receipt is None or set(tests).issubset(receipt.validation_receipt_ids) or succeeded),
        "required_proofs_present": not proofs
        or (receipt is not None and set(proofs).issubset(receipt.proof_receipt_ids)),
        "validation_receipts_current": receipt is not None and bool(receipt.validation_receipt_ids),
        "rollback_plan_bound": bool(plan.rollback_plan_id) if plan is not None else False,
        "no_protected_authority_mutation": bool(paths) and not protected,
        "paths_within_envelope": bool(paths) and not escaped,
        "repair_succeeded": succeeded,
        "isolated_worktree_when_required": worktree_ok,
        "predetermined_checks_bound": checks_bound or (succeeded and bool(tests)),
    }
    missing = set(LOW_RISK_MERGE_CONDITIONS) - set(conditions)
    if missing:
        raise RepairControllerError("low-risk merge conjunction is incomplete")
    return MappingProxyType({name: conditions[name] for name in LOW_RISK_MERGE_CONDITIONS})


def merge_disposition_for(
    *,
    request: RepairRequest,
    conditions: Mapping[str, bool],
    succeeded: bool,
) -> RepairMergeDisposition:
    """Map the conjunction onto the closed merge vocabulary."""

    risk = request.envelope.risk_assessment.risk_class
    if not succeeded:
        return RepairMergeDisposition.NOT_APPLICABLE
    if risk is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL:
        return RepairMergeDisposition.REJECTED
    if risk.rank >= RiskClass.R3_BOUNDED_REPOSITORY_MUTATION.rank:
        return RepairMergeDisposition.PROPOSAL_ONLY
    if all(conditions.get(name, False) for name in LOW_RISK_MERGE_CONDITIONS):
        return RepairMergeDisposition.AUTONOMOUS_MERGE_CANDIDATE
    return RepairMergeDisposition.PROPOSAL_ONLY


class AutonomousRepairController:
    """Tier-selection and scope facade over the existing repair engine.

    Effect execution, source mutation, merge, and completion remain owned by
    ``AutonomousRepairEngine``, ``AutonomousRepairMaterializer``,
    ``DecisionRuntime``, and the current merge authorities.
    """

    INTERFACE: Final[str] = AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        engine: AutonomousRepairEngine | None = None,
        materializer: AutonomousRepairMaterializer | None = None,
        model_call: Callable[[RepairRequest, AutonomousRepairPlan], Mapping[str, Any]]
        | None = None,
        backoff_milliseconds: int = DEFAULT_IDENTICAL_FAILURE_BACKOFF_MS,
        max_backoff_milliseconds: int = MAX_IDENTICAL_FAILURE_BACKOFF_MS,
        max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES,
    ) -> None:
        if engine is not None and not isinstance(engine, AutonomousRepairEngine):
            raise RepairControllerError(
                "engine must be the existing AutonomousRepairEngine"
            )
        if materializer is not None and not isinstance(
            materializer, AutonomousRepairMaterializer
        ):
            raise RepairControllerError(
                "materializer must be the existing AutonomousRepairMaterializer"
            )
        if engine is None:
            if repo_root is None:
                raise RepairControllerError("repo_root is required without an injected engine")
            engine = AutonomousRepairEngine(repo_root=repo_root)
        self._engine = engine
        self._materializer = materializer
        self._model_call = model_call
        self._backoff_milliseconds = _int(backoff_milliseconds, "backoff_milliseconds", minimum=1)
        self._max_backoff_milliseconds = _int(
            max_backoff_milliseconds, "max_backoff_milliseconds", minimum=1
        )
        self._max_identical_failures = _int(
            max_identical_failures, "max_identical_failures", minimum=1
        )
        self._failure_counts: dict[str, int] = {}
        self._failure_diagnostics: dict[str, Mapping[str, Any]] = {}
        self._model_call_count = 0

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine(self) -> AutonomousRepairEngine:
        return self._engine

    @property
    def materializer(self) -> AutonomousRepairMaterializer | None:
        return self._materializer

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    def tier_blockers(self, request: RepairRequest) -> tuple[str, ...]:
        """Return closed preconditions that block the requested tier."""

        preferred = request.preferred_tier
        if preferred not in _TIER_RANK:
            return ("unsupported_repair_tier",)
        if preferred is RepairTier.MODEL_ASSISTED_BOUNDED:
            blockers: list[str] = []
            if not request.worktree_id:
                blockers.append("isolated_worktree_required")
            if not request.effective_context_ids:
                blockers.append("context_references_required")
            if not request.effective_test_ids:
                blockers.append("predetermined_tests_required")
            return tuple(blockers)
        return ()

    def select_tier(self, request: RepairRequest) -> RepairTier:
        """Return the requested tier when admitted; never raise the ceiling."""

        blockers = self.tier_blockers(request)
        if blockers:
            raise RepairControllerError("tier_precondition_failed:" + ",".join(blockers))
        return request.preferred_tier

    def _suffix_reasons(self, request: RepairRequest) -> tuple[str, ...]:
        suffix = request.suffix_receipt
        if suffix is None:
            return ()
        if suffix.authorizes_effect or suffix.authorizes_full_replan:
            return ("suffix_receipt_authorizes_effect",)
        if suffix.objective_id != request.envelope.objective_id:
            return ("suffix_objective_mismatch",)
        if suffix.objective_revision != request.envelope.objective_revision:
            return ("suffix_revision_mismatch",)
        return ("suffix_contract_bound",)

    def admit(self, request: RepairRequest) -> RepairControllerResult:
        """Admit exact envelope/scope/tier without executing a mutation."""

        if not isinstance(request, RepairRequest):
            raise RepairControllerError("request must be a RepairRequest")
        suffix_reasons = self._suffix_reasons(request)
        if any(code != "suffix_contract_bound" for code in suffix_reasons):
            return self._rejected(request, suffix_reasons, selected_tier=None)
        risk = request.envelope.risk_assessment.risk_class
        needed_level = (
            AutonomyLevel.SELF_REPAIR_ISOLATED
            if risk.rank >= RiskClass.R3_BOUNDED_REPOSITORY_MUTATION.rank
            else AutonomyLevel.EXECUTE_REVERSIBLE
        )
        if (
            request.envelope.autonomy_level.rank < needed_level.rank
            or not request.policy.allows(needed_level, risk)
        ):
            return self._rejected(
                request,
                ("autonomy_level_denied",),
                selected_tier=request.preferred_tier,
            )
        scope = _scope_rejection(request.predicted_files, request.envelope.allowed_paths)
        if scope is not None:
            return self._rejected(
                request,
                (scope, "predicted_files_rejected"),
                selected_tier=request.preferred_tier,
            )
        colliding = [
            symbol
            for symbol in request.predicted_symbols
            if symbol not in request.envelope.allowed_symbols
            and request.envelope.allowed_symbols
        ]
        if colliding:
            return self._rejected(
                request,
                ("symbol_escape",),
                selected_tier=request.preferred_tier,
            )
        blockers = self.tier_blockers(request)
        if blockers:
            return self._rejected(
                request,
                ("tier_precondition_failed", *blockers),
                selected_tier=request.preferred_tier,
            )
        tier = self.select_tier(request)
        plan = self._plan_for(request, tier)
        return RepairControllerResult(
            disposition=RepairControllerDisposition.ADMITTED,
            merge_disposition=RepairMergeDisposition.NOT_APPLICABLE,
            reason_codes=("admitted", tier.value, *suffix_reasons),
            selected_tier=tier,
            plan=plan,
        )

    def run(self, request: RepairRequest) -> RepairControllerResult:
        """Admit, optionally invoke the existing engine, and bind merge disposition."""

        admitted = self.admit(request)
        if admitted.disposition is RepairControllerDisposition.REJECTED:
            return admitted
        assert admitted.plan is not None
        plan = admitted.plan
        model_calls = 0
        diagnostic_reused = False
        backoff_ms = 0
        extra_reasons: list[str] = list(admitted.reason_codes)

        if plan.repair_tier is RepairTier.MODEL_ASSISTED_BOUNDED:
            signature = request.failure_signature or content_identity(
                {
                    "files": list(request.predicted_files),
                    "symbols": list(request.predicted_symbols),
                    "tier": plan.repair_tier.value,
                    "envelope_id": request.envelope.envelope_id,
                }
            )
            seen = self._failure_counts.get(signature, 0)
            if seen > 0:
                diagnostic_reused = True
                self._failure_counts[signature] = seen + 1
                backoff_ms = min(
                    self._backoff_milliseconds * (2 ** (seen - 1)),
                    self._max_backoff_milliseconds,
                )
                extra_reasons.extend(("identical_failure_backoff", "diagnostic_reused"))
                if seen + 1 >= self._max_identical_failures:
                    extra_reasons.append("identical_failure_exhausted")
                receipt = self._receipt_for(
                    request,
                    plan,
                    terminal=TerminalStatus.BLOCKED,
                    diagnostic_receipt_id=request.diagnostic_receipt_id
                    or str(self._failure_diagnostics.get(signature, {}).get("diagnostic_receipt_id") or ""),
                    failure_signature=signature,
                )
                conditions = evaluate_low_risk_merge_conjunction(
                    request=request, plan=plan, receipt=receipt
                )
                return RepairControllerResult(
                    disposition=RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF,
                    merge_disposition=RepairMergeDisposition.NOT_APPLICABLE,
                    reason_codes=tuple(extra_reasons),
                    selected_tier=plan.repair_tier,
                    plan=plan,
                    receipt=receipt,
                    low_risk_conditions=conditions,
                    diagnostic_reused=True,
                    backoff_milliseconds=backoff_ms,
                    model_call_count=0,
                )
            if self._model_call is None:
                return self._rejected(
                    request,
                    ("model_route_unavailable",),
                    selected_tier=plan.repair_tier,
                    plan=plan,
                )
            diagnostic = self._model_call(request, plan)
            if not isinstance(diagnostic, Mapping):
                raise RepairControllerError("model_call must return a mapping")
            _reject_forbidden_keys(diagnostic, "model_diagnostic")
            self._model_call_count += 1
            model_calls = 1
            self._failure_counts[signature] = 1
            self._failure_diagnostics[signature] = MappingProxyType(dict(diagnostic))
            extra_reasons.append("model_assisted_invoked")

        engine_report: Mapping[str, Any] | None = None
        engine_invoked = False
        if request.work_items:
            report = self._engine.run(request.work_items)
            engine_invoked = True
            engine_report = report.to_dict() if hasattr(report, "to_dict") else dict(report)
            extra_reasons.append("existing_engine_invoked")
            engine_calls = int(engine_report.get("model_call_count") or 0)
            if plan.repair_tier is not RepairTier.MODEL_ASSISTED_BOUNDED and engine_calls:
                return self._rejected(
                    request,
                    ("engine_model_call_forbidden",),
                    selected_tier=plan.repair_tier,
                    plan=plan,
                    engine_report=engine_report,
                    engine_invoked=True,
                )

        materialize_report: Mapping[str, Any] | None = None
        mutation_applied = False
        if (
            request.allow_code_edit_materialize
            or request.source_edit_operator is not None
            or request.edit_plan is not None
        ):
            source = self.admit_source_edit(request, plan=plan)
            if source.disposition is RepairControllerDisposition.REJECTED:
                return source
            materialize_report = source.materialize_report
            mutation_applied = source.mutation_applied
            extra_reasons.extend(code for code in source.reason_codes if code not in extra_reasons)

        succeeded = bool(request.validation_receipt_ids) and not mutation_applied
        terminal = TerminalStatus.SUCCEEDED if succeeded else TerminalStatus.PENDING
        if mutation_applied:
            extra_reasons.append("source_edit_validation_pending")
        receipt = self._receipt_for(request, plan, terminal=terminal)
        if terminal is TerminalStatus.SUCCEEDED and not request.validation_receipt_ids:
            raise RepairControllerError("successful repair requires current validation evidence")
        conditions = evaluate_low_risk_merge_conjunction(
            request=request,
            plan=plan,
            receipt=receipt,
            changed_paths=receipt.changed_paths or plan.predicted_files,
        )
        merge = merge_disposition_for(
            request=request,
            conditions=conditions,
            succeeded=terminal is TerminalStatus.SUCCEEDED,
        )
        extra_reasons.append(merge.value)
        return RepairControllerResult(
            disposition=RepairControllerDisposition.EXECUTED,
            merge_disposition=merge,
            reason_codes=tuple(dict.fromkeys(extra_reasons)),
            selected_tier=plan.repair_tier,
            plan=plan,
            receipt=receipt,
            engine_report=engine_report,
            materialize_report=materialize_report,
            low_risk_conditions=conditions,
            diagnostic_reused=diagnostic_reused,
            backoff_milliseconds=backoff_ms,
            model_call_count=model_calls,
            engine_invoked=engine_invoked,
            mutation_applied=mutation_applied,
        )

    def admit_source_edit(
        self,
        request: RepairRequest,
        *,
        plan: AutonomousRepairPlan | None = None,
    ) -> RepairControllerResult:
        """Admit an exact source-edit operator through the existing materializer.

        A policy flag cannot become source-edit admission.  Catalog or
        analysis-only rows never count as applied.  This facade never writes
        bytes itself.
        """

        if not isinstance(request, RepairRequest):
            raise RepairControllerError("request must be a RepairRequest")
        if plan is None:
            admitted = self.admit(request)
            if admitted.disposition is RepairControllerDisposition.REJECTED:
                return admitted
            plan = admitted.plan
        assert plan is not None
        reasons: list[str] = ["existing_source_edit_operator"]
        if request.allow_code_edit_materialize and request.source_edit_operator is None:
            return self._rejected(
                request,
                ("policy_flag_is_not_source_edit_admission",),
                selected_tier=plan.repair_tier,
                plan=plan,
            )
        operator_raw = request.source_edit_operator
        if operator_raw is None:
            return self._rejected(
                request,
                ("typed_admitted_source_edit_operator_required",),
                selected_tier=plan.repair_tier,
                plan=plan,
            )
        try:
            operator = AdmittedSourceEditOperator.from_mapping(operator_raw)
        except AdmittedSourceEditError as exc:
            detail = str(exc).split(":", 1)[0].replace(" ", "_")
            return self._rejected(
                request,
                ("source_edit_operator_not_admitted", detail),
                selected_tier=plan.repair_tier,
                plan=plan,
            )
        target = operator.relative_path
        scope = _scope_rejection((target,), request.envelope.allowed_paths)
        if scope is not None:
            return self._rejected(
                request,
                (scope, "source_edit_path_rejected"),
                selected_tier=plan.repair_tier,
                plan=plan,
            )
        if not path_within_allowed(target, plan.allowed_paths):
            return self._rejected(
                request,
                ("scope_escape", "source_edit_path_rejected"),
                selected_tier=plan.repair_tier,
                plan=plan,
            )
        if not path_within_allowed(target, request.predicted_files):
            return self._rejected(
                request,
                ("scope_escape", "source_edit_path_rejected"),
                selected_tier=plan.repair_tier,
                plan=plan,
            )
        if self._materializer is None:
            return RepairControllerResult(
                disposition=RepairControllerDisposition.ADMITTED,
                merge_disposition=RepairMergeDisposition.NOT_APPLICABLE,
                reason_codes=("source_edit_admitted", "materializer_not_bound", *reasons),
                selected_tier=plan.repair_tier,
                plan=plan,
            )
        edit_plan = dict(request.edit_plan or {})
        edit_plan.setdefault("plan_id", plan.plan_id)
        edit_plan.setdefault("work_id", request.envelope.task_id)
        edit_plan.setdefault("operation", request.predicted_symbols[0])
        edit_plan["materialize_ready"] = True
        edit_plan["preferred_path"] = target
        edit_plan["source_edit_operator"] = dict(operator_raw)
        report = self._materializer.materialize_plans([edit_plan])
        receipts = list(report.get("receipts") or ())
        first = receipts[0] if receipts else {}
        mutation_applied = bool(first.get("mutation_applied"))
        status = str(first.get("status") or report.get("status") or "")
        if status == "rejected" or (report.get("passed") is False and not mutation_applied):
            if not mutation_applied:
                materialize_reasons = tuple(
                    str(item).replace(" ", "_")
                    for item in (first.get("reasons") or ())
                    if str(item).replace(" ", "_")
                )
                return RepairControllerResult(
                    disposition=RepairControllerDisposition.REJECTED,
                    merge_disposition=RepairMergeDisposition.REJECTED,
                    reason_codes=(
                        "source_edit_materialize_rejected",
                        *materialize_reasons,
                    ),
                    selected_tier=plan.repair_tier,
                    plan=plan,
                    materialize_report=report,
                    mutation_applied=False,
                )
        extra = ["existing_materializer_invoked"]
        if mutation_applied:
            extra.append("source_edit_validation_pending")
        receipt = self._receipt_for(
            request,
            plan,
            terminal=TerminalStatus.PENDING if mutation_applied else TerminalStatus.BLOCKED,
            changed_paths=(target,) if mutation_applied else (),
        )
        return RepairControllerResult(
            disposition=RepairControllerDisposition.EXECUTED,
            merge_disposition=RepairMergeDisposition.NOT_APPLICABLE,
            reason_codes=tuple(dict.fromkeys((*reasons, *extra))),
            selected_tier=plan.repair_tier,
            plan=plan,
            receipt=receipt,
            materialize_report=report,
            mutation_applied=mutation_applied,
        )

    def _plan_for(self, request: RepairRequest, tier: RepairTier) -> AutonomousRepairPlan:
        return AutonomousRepairPlan(
            objective_id=request.envelope.objective_id,
            task_id=request.envelope.task_id,
            repair_tier=tier,
            predicted_files=request.predicted_files,
            predicted_symbols=request.predicted_symbols,
            patch_envelope_id=request.envelope.envelope_id,
            context_reference_ids=request.effective_context_ids,
            required_test_ids=request.effective_test_ids,
            required_proof_ids=request.effective_proof_ids,
            worktree_id=request.worktree_id
            or ("worktree:deterministic" if tier is not RepairTier.MODEL_ASSISTED_BOUNDED else ""),
            allowed_paths=request.envelope.allowed_paths,
            forbidden_symbols=request.forbidden_symbols,
            rollback_plan_id=request.rollback_plan_id,
            risk_class=request.envelope.risk_assessment.risk_class,
            max_changed_files=request.max_changed_files,
            max_changed_lines=request.max_changed_lines,
        )

    def _receipt_for(
        self,
        request: RepairRequest,
        plan: AutonomousRepairPlan,
        *,
        terminal: TerminalStatus,
        changed_paths: Sequence[str] | None = None,
        diagnostic_receipt_id: str = "",
        failure_signature: str = "",
    ) -> AutonomousRepairReceipt:
        paths = tuple(changed_paths) if changed_paths is not None else plan.predicted_files
        validation_ids = request.validation_receipt_ids
        if terminal is TerminalStatus.SUCCEEDED and not validation_ids:
            validation_ids = request.effective_test_ids
        return AutonomousRepairReceipt(
            plan_id=plan.plan_id,
            envelope_id=request.envelope.envelope_id,
            terminal_status=terminal,
            changed_paths=paths if terminal is not TerminalStatus.BLOCKED else (),
            validation_receipt_ids=validation_ids,
            proof_receipt_ids=request.proof_receipt_ids,
            adversarial_assurance_receipt_ids=request.adversarial_assurance_receipt_ids,
            rollback_receipt_id=plan.rollback_plan_id
            if terminal in {TerminalStatus.FAILED, TerminalStatus.BLOCKED}
            else "",
            failure_signature=failure_signature or request.failure_signature,
            diagnostic_receipt_id=diagnostic_receipt_id or request.diagnostic_receipt_id,
            authorizes_merge=False,
        )

    def _rejected(
        self,
        request: RepairRequest,
        reasons: Sequence[str],
        *,
        selected_tier: RepairTier | None,
        plan: AutonomousRepairPlan | None = None,
        engine_report: Mapping[str, Any] | None = None,
        engine_invoked: bool = False,
    ) -> RepairControllerResult:
        return RepairControllerResult(
            disposition=RepairControllerDisposition.REJECTED,
            merge_disposition=RepairMergeDisposition.REJECTED,
            reason_codes=tuple(reasons),
            selected_tier=selected_tier,
            plan=plan,
            engine_report=engine_report,
            engine_invoked=engine_invoked,
        )

    def snapshot(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema": REPAIR_CONTROLLER_SNAPSHOT_SCHEMA,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "engine_type": type(self._engine).__name__,
            "engine_module": type(self._engine).__module__,
            "model_call_count": self._model_call_count,
            "failure_counts": dict(sorted(self._failure_counts.items())),
            "failure_diagnostics": {
                key: dict(value) for key, value in sorted(self._failure_diagnostics.items())
            },
            "backoff_milliseconds": self._backoff_milliseconds,
            "max_backoff_milliseconds": self._max_backoff_milliseconds,
            "max_identical_failures": self._max_identical_failures,
        }
        payload["snapshot_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "snapshot_id"}
        )
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > MAX_REPAIR_CONTROLLER_SNAPSHOT_BYTES:
            raise RepairControllerError("repair-controller snapshot exceeds its bounded size")
        return MappingProxyType(payload)


__all__ = [
    "AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE",
    "DEFAULT_IDENTICAL_FAILURE_BACKOFF_MS",
    "LOW_RISK_MERGE_CONDITIONS",
    "MAX_IDENTICAL_FAILURE_BACKOFF_MS",
    "PROTECTED_AUTHORITY_PREFIXES",
    "REPAIR_CONTROLLER_RESULT_SCHEMA",
    "REPAIR_CONTROLLER_SNAPSHOT_SCHEMA",
    "SELF_EDIT_PATH",
    "AutonomousRepairController",
    "RepairControllerDisposition",
    "RepairControllerError",
    "RepairControllerResult",
    "RepairMergeDisposition",
    "RepairRequest",
    "evaluate_low_risk_merge_conjunction",
    "merge_disposition_for",
    "path_within_allowed",
    "protected_authority_reason",
]
