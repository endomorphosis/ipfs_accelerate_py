# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded facade over the existing autonomous-repair engine.

``AutonomousRepairController@1`` selects among deterministic, template-
constrained, and model-assisted tiers and binds an exact envelope, isolated
worktree, predetermined checks, identical-failure backoff, and merge
disposition.  It is not a second repair engine, materializer, worktree
manager, DecisionRuntime, or merge authority.

Admission and execution remain owned by
``agent_supervisor.autonomous_repair``.  Source-byte mutation is admitted
only by ``AdmittedSourceEditOperator``; a policy flag is not admission.
Repair receipts stay evidence: they never independently authorize merge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Any, ClassVar, Final

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
AUTONOMOUS_REPAIR_CONTROLLER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/autonomous-repair-controller@1"
)
REPAIR_CONTROLLER_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-request@1"
)
REPAIR_CONTROLLER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-result@1"
)
REPAIR_MERGE_EVALUATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-merge-evaluation@1"
)
REPAIR_FAILURE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-failure-record@1"
)

SELF_EDIT_PATH: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
)
DEFAULT_BASE_BACKOFF_MS: Final[int] = 10
DEFAULT_MAX_BACKOFF_MS: Final[int] = 40
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 4
MAX_FAILURE_RECORDS: Final[int] = 1_024
MAX_CHANGED_LINE_COUNT: Final[int] = 1_000_000

# Operator-protected authority surfaces plus validator / policy-key files.
# Exact-file matches are rejected even when an envelope otherwise allows them.
PROTECTED_AUTHORITY_PATHS: Final[frozenset[str]] = frozenset(
    {
        ".gitignore",
        "docs/architecture/AGENT_SUPERVISOR_AUTONOMOUS_META_CONTROLLER_PLAN.md",
        "docs/architecture/agent_supervisor_autonomous_meta_controller.objectives.md",
        "docs/architecture/agent_supervisor_autonomous_meta_controller.todo.md",
        "config/agent_supervisor_autonomous_meta_controller_scheduler.json",
        "scripts/validate_agent_supervisor_autonomous_meta_controller_board.py",
        "scripts/materialize_agent_supervisor_autonomous_meta_controller_board.py",
        "scripts/ops/agent_supervisor/quack_state_server.py",
        "scripts/lgswf_start_quack_control.py",
        "ipfs_accelerate_py/agent_implementation_route.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_catalog.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/mcp_invocation_trace.py",
        "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
        "ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "ipfs_accelerate_py/llm_router.py",
        SELF_EDIT_PATH,
    }
)

_VALIDATOR_POLICY_KEY_PATH_MARKERS: Final[tuple[str, ...]] = (
    "authority_policy",
    "llm_router",
    "profile_authority",
    "project_dependency_preflight",
    "promotion_admission",
    "promotion_pointer",
    "proposal_validation",
    "trusted_key",
    "validator_policy",
)
_VALIDATOR_POLICY_KEY_SYMBOL_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "authority_policy",
        "promotion_rules",
        "trusted_key",
        "trusted_keys",
        "validator_policy_key",
        "validator_policy_keys",
    }
)

# Every named condition must hold before R2 autonomous merge is eligible.
LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "autonomous_merge_enabled",
    "risk_class_r2_or_lower",
    "reversible",
    "autonomy_level_execute_reversible",
    "policy_allows_level",
    "exact_paths_bound",
    "exact_symbols_bound",
    "changed_paths_within_predicted",
    "predicted_within_allowed",
    "no_protected_authority_paths",
    "no_self_edit",
    "no_validator_policy_key_mutation",
    "predetermined_tests_satisfied",
    "predetermined_proofs_satisfied",
    "rollback_plan_bound",
    "isolated_worktree_if_model_assisted",
    "patch_bounds_respected",
    "terminal_succeeded",
    "not_security_or_protocol_sensitive",
    "not_irreversible_or_legal",
    "validation_receipts_present",
)

_TIER_RANK: Final[Mapping[RepairTier, int]] = {
    RepairTier.DETERMINISTIC: 0,
    RepairTier.TEMPLATE_CONSTRAINED: 1,
    RepairTier.MODEL_ASSISTED_BOUNDED: 2,
}
_LOW_RISK_CLASSES: Final[frozenset[RiskClass]] = frozenset(
    {
        RiskClass.R0_PURE,
        RiskClass.R1_READ_ONLY,
        RiskClass.R2_REVERSIBLE_LOCAL,
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
    """Closed outcome of one facade invocation.  Never a merge grant."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    BACKED_OFF = "backed_off"
    EXHAUSTED = "exhausted"
    ROLLED_BACK = "rolled_back"


class RepairMergeDisposition(str, Enum):
    """Closed merge recommendation.  Never independent merge authority."""

    NOT_CONSIDERED = "not_considered"
    PROPOSAL_ONLY = "proposal_only"
    AUTONOMOUS_MERGE_ELIGIBLE = "autonomous_merge_eligible"


class RepairMutationClass(str, Enum):
    """Closed classification of a proposed path/symbol mutation."""

    ADMITTED_SCOPE = "admitted_scope"
    SCOPE_ESCAPE = "scope_escape"
    SELF_EDIT = "self_edit"
    VALIDATOR_POLICY_KEY = "validator_policy_key"


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
    preserve_order: bool = False,
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


def _posix_path(value: Any, name: str) -> str:
    result = _identifier(value, name)
    if "\\" in result:
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    parsed = PurePosixPath(result)
    if parsed.is_absolute() or ".." in parsed.parts or result in {".", ""}:
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    return parsed.as_posix()


def _posix_paths(
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
    return tuple(sorted(normalized))


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


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise RepairControllerError(f"{name} must be an integer between {minimum} and {maximum}")
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
            raise RepairControllerError(f"{name} contains forbidden private or executable data")


def path_in_prefixes(path: str, prefixes: Sequence[str]) -> bool:
    """Return True when ``path`` equals or is nested under any prefix."""

    candidate = PurePosixPath(path).as_posix()
    for prefix in prefixes:
        bound = PurePosixPath(prefix).as_posix().rstrip("/")
        if candidate == bound or candidate.startswith(bound + "/"):
            return True
    return False


def classify_mutation(
    path: str,
    *,
    allowed_paths: Sequence[str],
    forbidden_symbols: Sequence[str] = (),
    symbols: Sequence[str] = (),
) -> RepairMutationClass:
    """Classify one proposed mutation against envelope and authority paths."""

    relative = _posix_path(path, "path")
    if relative == SELF_EDIT_PATH or path_in_prefixes(relative, (SELF_EDIT_PATH,)):
        return RepairMutationClass.SELF_EDIT
    if _is_validator_policy_key_path(relative) or _symbols_are_policy_keys(
        symbols, forbidden_symbols
    ):
        return RepairMutationClass.VALIDATOR_POLICY_KEY
    if not path_in_prefixes(relative, allowed_paths):
        return RepairMutationClass.SCOPE_ESCAPE
    return RepairMutationClass.ADMITTED_SCOPE


def _is_validator_policy_key_path(path: str) -> bool:
    if path in PROTECTED_AUTHORITY_PATHS:
        return path != SELF_EDIT_PATH
    lowered = path.lower().replace("-", "_")
    return any(marker in lowered for marker in _VALIDATOR_POLICY_KEY_PATH_MARKERS)


def _symbols_are_policy_keys(
    symbols: Sequence[str],
    forbidden_symbols: Sequence[str],
) -> bool:
    forbidden = {item.lower().replace("-", "_") for item in forbidden_symbols}
    for symbol in symbols:
        normalized = symbol.lower().replace("-", "_")
        if normalized in forbidden or normalized in _VALIDATOR_POLICY_KEY_SYMBOL_MARKERS:
            return True
        if any(marker in normalized for marker in _VALIDATOR_POLICY_KEY_SYMBOL_MARKERS):
            return True
    return False


def _tier_rank(tier: RepairTier) -> int:
    return _TIER_RANK[tier]


def select_repair_tier(
    *,
    requested: RepairTier,
    deterministic_applicable: bool,
    template_applicable: bool,
    model_assistance_requested: bool,
) -> RepairTier:
    """Select the cheapest admissible tier; never raise above ``requested``."""

    if deterministic_applicable:
        selected = RepairTier.DETERMINISTIC
    elif template_applicable:
        selected = RepairTier.TEMPLATE_CONSTRAINED
    elif model_assistance_requested:
        selected = RepairTier.MODEL_ASSISTED_BOUNDED
    else:
        selected = requested
    if _tier_rank(selected) > _tier_rank(requested):
        return requested
    return selected


def repair_failure_signature(
    plan: AutonomousRepairPlan,
    *,
    diagnostic_receipt_id: str = "",
    extra: str = "",
) -> str:
    """Content identity of one identical-failure class.  Never a transcript."""

    payload = {
        "schema": REPAIR_FAILURE_RECORD_SCHEMA,
        "objective_id": plan.objective_id,
        "task_id": plan.task_id,
        "predicted_files": list(plan.predicted_files),
        "predicted_symbols": list(plan.predicted_symbols),
        "diagnostic_receipt_id": diagnostic_receipt_id,
        "extra": extra,
    }
    return content_identity(payload)


def evaluate_merge_conjunction(
    *,
    policy: AutonomyPolicy,
    envelope: AutonomyEnvelope,
    plan: AutonomousRepairPlan,
    selected_tier: RepairTier,
    changed_paths: Sequence[str],
    changed_line_count: int,
    validation_receipt_ids: Sequence[str],
    proof_receipt_ids: Sequence[str],
    terminal_status: TerminalStatus,
    isolated_worktree: bool,
) -> Mapping[str, bool]:
    """Evaluate every stated low-risk merge condition.  Conjunction is exact."""

    risk = envelope.risk_assessment
    predicted = tuple(plan.predicted_files)
    changed = tuple(changed_paths)
    allowed = tuple(envelope.allowed_paths) or tuple(plan.allowed_paths)
    mutation_classes = [
        classify_mutation(
            path,
            allowed_paths=allowed,
            forbidden_symbols=plan.forbidden_symbols,
            symbols=plan.predicted_symbols,
        )
        for path in (*predicted, *changed)
    ]
    required_tests = tuple(envelope.required_test_ids or plan.required_test_ids)
    tests_ok = True
    if required_tests:
        tests_ok = set(required_tests).issubset(validation_receipt_ids)
        if envelope.required_test_ids:
            tests_ok = tests_ok and set(envelope.required_test_ids).issubset(plan.required_test_ids)
    required_proofs = tuple(envelope.required_proof_ids or plan.required_proof_ids)
    proofs_ok = True
    if required_proofs:
        proofs_ok = set(required_proofs).issubset(proof_receipt_ids)
        if envelope.required_proof_ids:
            proofs_ok = proofs_ok and set(envelope.required_proof_ids).issubset(
                plan.required_proof_ids
            )
    isolated_ok = (
        selected_tier is not RepairTier.MODEL_ASSISTED_BOUNDED
        or (bool(plan.worktree_id) and isolated_worktree)
    )
    conditions = {
        "autonomous_merge_enabled": policy.autonomous_merge_enabled,
        "risk_class_r2_or_lower": risk.risk_class in _LOW_RISK_CLASSES,
        "reversible": envelope.reversible and risk.reversible,
        "autonomy_level_execute_reversible": (
            envelope.autonomy_level.rank >= AutonomyLevel.EXECUTE_REVERSIBLE.rank
        ),
        "policy_allows_level": policy.allows(envelope.autonomy_level, risk.risk_class),
        "exact_paths_bound": bool(predicted) and all(path_in_prefixes(path, allowed) for path in predicted),
        "exact_symbols_bound": bool(plan.predicted_symbols)
        and (
            not envelope.allowed_symbols
            or set(plan.predicted_symbols).issubset(envelope.allowed_symbols)
        ),
        "changed_paths_within_predicted": bool(changed)
        and all(path_in_prefixes(path, predicted) for path in changed),
        "predicted_within_allowed": all(path_in_prefixes(path, allowed) for path in predicted),
        "no_protected_authority_paths": all(
            item is RepairMutationClass.ADMITTED_SCOPE for item in mutation_classes
        ),
        "no_self_edit": all(item is not RepairMutationClass.SELF_EDIT for item in mutation_classes),
        "no_validator_policy_key_mutation": all(
            item is not RepairMutationClass.VALIDATOR_POLICY_KEY for item in mutation_classes
        ),
        "predetermined_tests_satisfied": tests_ok,
        "predetermined_proofs_satisfied": proofs_ok,
        "rollback_plan_bound": bool(plan.rollback_plan_id),
        "isolated_worktree_if_model_assisted": isolated_ok,
        "patch_bounds_respected": (
            len(set(changed) or set(predicted)) <= plan.max_changed_files
            and changed_line_count <= plan.max_changed_lines
            and changed_line_count >= 0
        ),
        "terminal_succeeded": terminal_status is TerminalStatus.SUCCEEDED,
        "not_security_or_protocol_sensitive": (
            not risk.security_sensitive and not risk.protocol_sensitive
        ),
        "not_irreversible_or_legal": (
            not risk.irreversible_external_effect and not risk.legal_or_financial_effect
        ),
        "validation_receipts_present": bool(validation_receipt_ids),
    }
    if set(conditions) != set(LOW_RISK_MERGE_CONDITIONS):
        raise RepairControllerError("merge conjunction must report every stated low-risk condition")
    return MappingProxyType({name: bool(conditions[name]) for name in LOW_RISK_MERGE_CONDITIONS})


@dataclass(frozen=True)
class RepairFailureRecord:
    """One bounded identical-failure observation.  Diagnosis is referenced."""

    signature: str
    diagnostic_receipt_id: str = ""
    count: int = 1
    backoff_milliseconds: int = 0
    model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "signature", _identifier(self.signature, "signature"))
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _identifier(self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False),
        )
        object.__setattr__(self, "count", _int(self.count, "count", minimum=1))
        object.__setattr__(
            self,
            "backoff_milliseconds",
            _int(self.backoff_milliseconds, "backoff_milliseconds"),
        )
        object.__setattr__(self, "model_calls", _int(self.model_calls, "model_calls"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_FAILURE_RECORD_SCHEMA,
            "signature": self.signature,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "count": self.count,
            "backoff_milliseconds": self.backoff_milliseconds,
            "model_calls": self.model_calls,
        }


@dataclass(frozen=True)
class RepairMergeEvaluation:
    """Exact conjunction of the stated low-risk autonomous-merge conditions."""

    disposition: RepairMergeDisposition
    conditions: Mapping[str, bool]
    unsatisfied: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairMergeDisposition, "disposition"),
        )
        if not isinstance(self.conditions, Mapping):
            raise RepairControllerError("conditions must be a mapping")
        if set(self.conditions) != set(LOW_RISK_MERGE_CONDITIONS):
            raise RepairControllerError("conditions must cover every stated low-risk condition")
        frozen = {name: _bool(self.conditions[name], name) for name in LOW_RISK_MERGE_CONDITIONS}
        object.__setattr__(self, "conditions", MappingProxyType(frozen))
        expected = tuple(name for name, ok in frozen.items() if not ok)
        object.__setattr__(self, "unsatisfied", _identifiers(self.unsatisfied, "unsatisfied"))
        if self.unsatisfied != expected:
            raise RepairControllerError("unsatisfied merge conditions are inconsistent")
        if (
            self.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
            and self.unsatisfied
        ):
            raise RepairControllerError(
                "autonomous merge requires every stated low-risk condition"
            )

    @property
    def authorizes_merge(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_MERGE_EVALUATION_SCHEMA,
            "disposition": self.disposition.value,
            "conditions": dict(self.conditions),
            "unsatisfied": list(self.unsatisfied),
            "authorizes_merge": False,
        }

    @classmethod
    def from_conditions(
        cls,
        conditions: Mapping[str, bool],
        *,
        considered: bool,
        risk_class: RiskClass,
    ) -> RepairMergeEvaluation:
        ordered = {name: bool(conditions[name]) for name in LOW_RISK_MERGE_CONDITIONS}
        unsatisfied = tuple(name for name, ok in ordered.items() if not ok)
        if not considered:
            disposition = RepairMergeDisposition.NOT_CONSIDERED
        elif not unsatisfied and risk_class in _LOW_RISK_CLASSES:
            disposition = RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
        else:
            disposition = RepairMergeDisposition.PROPOSAL_ONLY
        return cls(disposition=disposition, conditions=ordered, unsatisfied=unsatisfied)


@dataclass(frozen=True)
class RepairControllerRequest:
    """One envelope-bound repair attempt.  Never a raw prompt or source body."""

    plan: AutonomousRepairPlan
    envelope: AutonomyEnvelope
    policy: AutonomyPolicy
    changed_paths: tuple[str, ...] = ()
    changed_line_count: int = 0
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    diagnostic_receipt_id: str = ""
    failure_extra: str = ""
    source_edit_operator: Mapping[str, Any] | None = None
    suffix_receipt: PlanSuffixInvalidationReceipt | None = None
    deterministic_applicable: bool = True
    template_applicable: bool = False
    model_assistance_requested: bool = False
    isolated_worktree: bool = False
    roll_back: bool = False
    apply_source_edit: bool = False
    delegate_to_engine: bool = False
    now_ms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan")
        if not isinstance(self.envelope, AutonomyEnvelope):
            raise RepairControllerError("envelope must be an AutonomyEnvelope")
        if not isinstance(self.policy, AutonomyPolicy):
            raise RepairControllerError("policy must be an AutonomyPolicy")
        if self.envelope.policy_id != self.policy.policy_id:
            raise RepairControllerError("envelope policy_id does not match policy")
        if self.plan.patch_envelope_id != self.envelope.envelope_id:
            raise RepairControllerError("repair plan is not bound to the envelope")
        if self.plan.objective_id != self.envelope.objective_id:
            raise RepairControllerError("repair plan objective_id does not match envelope")
        if self.plan.task_id != self.envelope.task_id:
            raise RepairControllerError("repair plan task_id does not match envelope")
        object.__setattr__(
            self, "changed_paths", _posix_paths(self.changed_paths, "changed_paths")
        )
        object.__setattr__(
            self,
            "changed_line_count",
            _int(self.changed_line_count, "changed_line_count", maximum=MAX_CHANGED_LINE_COUNT),
        )
        for name in (
            "validation_receipt_ids",
            "proof_receipt_ids",
            "adversarial_assurance_receipt_ids",
        ):
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _identifier(self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False),
        )
        object.__setattr__(
            self,
            "failure_extra",
            _identifier(self.failure_extra, "failure_extra", required=False),
        )
        if self.source_edit_operator is not None:
            if not isinstance(self.source_edit_operator, Mapping):
                raise RepairControllerError("source_edit_operator must be an object")
            if len(self.source_edit_operator) > MAX_MAPPING_ITEMS:
                raise RepairControllerError("source_edit_operator contains too many items")
            _reject_forbidden_keys(self.source_edit_operator, "source_edit_operator")
            object.__setattr__(
                self,
                "source_edit_operator",
                MappingProxyType(dict(self.source_edit_operator)),
            )
        if self.suffix_receipt is not None and not isinstance(
            self.suffix_receipt, PlanSuffixInvalidationReceipt
        ):
            raise RepairControllerError("suffix_receipt must be a PlanSuffixInvalidationReceipt")
        for name in (
            "deterministic_applicable",
            "template_applicable",
            "model_assistance_requested",
            "isolated_worktree",
            "roll_back",
            "apply_source_edit",
            "delegate_to_engine",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(self, "now_ms", _int(self.now_ms, "now_ms"))
        if self.suffix_receipt is not None:
            if self.suffix_receipt.objective_id != self.envelope.objective_id:
                raise RepairControllerError("suffix receipt objective_id does not match envelope")
            if self.suffix_receipt.objective_revision != self.envelope.objective_revision:
                raise RepairControllerError(
                    "suffix receipt objective_revision does not match envelope"
                )

    @property
    def failure_signature(self) -> str:
        return repair_failure_signature(
            self.plan,
            diagnostic_receipt_id=self.diagnostic_receipt_id,
            extra=self.failure_extra,
        )


@dataclass(frozen=True)
class RepairControllerResult:
    """Facade outcome.  Evidence only; never an effect or merge permit."""

    disposition: RepairControllerDisposition
    selected_tier: RepairTier
    reason_codes: tuple[str, ...]
    merge: RepairMergeEvaluation
    plan: AutonomousRepairPlan
    envelope_id: str
    failure_signature: str
    mutation_class: RepairMutationClass = RepairMutationClass.ADMITTED_SCOPE
    receipt: AutonomousRepairReceipt | None = None
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    model_call_count: int = 0
    source_edit_admitted: bool = False
    engine_invoked: bool = False
    materializer_invoked: bool = False
    suffix_receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairControllerDisposition, "disposition"),
        )
        object.__setattr__(
            self, "selected_tier", _enum(self.selected_tier, RepairTier, "selected_tier")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", required=True, preserve_order=True),
        )
        if not isinstance(self.merge, RepairMergeEvaluation):
            raise RepairControllerError("merge must be a RepairMergeEvaluation")
        if not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan")
        object.__setattr__(self, "envelope_id", _identifier(self.envelope_id, "envelope_id"))
        object.__setattr__(
            self, "failure_signature", _identifier(self.failure_signature, "failure_signature")
        )
        object.__setattr__(
            self,
            "mutation_class",
            _enum(self.mutation_class, RepairMutationClass, "mutation_class"),
        )
        if self.receipt is not None and not isinstance(self.receipt, AutonomousRepairReceipt):
            raise RepairControllerError("receipt must be an AutonomousRepairReceipt")
        if self.receipt is not None and self.receipt.authorizes_merge:
            raise RepairControllerError("repair receipts cannot independently authorize merge")
        for name in (
            "diagnostic_reused",
            "source_edit_admitted",
            "engine_invoked",
            "materializer_invoked",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "backoff_milliseconds",
            _int(self.backoff_milliseconds, "backoff_milliseconds"),
        )
        object.__setattr__(self, "model_call_count", _int(self.model_call_count, "model_call_count"))
        object.__setattr__(
            self,
            "suffix_receipt_id",
            _identifier(self.suffix_receipt_id, "suffix_receipt_id", required=False),
        )
        encoded = canonical_json(self.to_dict(include_identity=False)).encode("utf-8")
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise RepairControllerError("repair-controller result exceeds its bounded size")

    @property
    def result_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def authorizes_effect(self) -> bool:
        return False

    @property
    def authorizes_merge(self) -> bool:
        return False

    @property
    def admitted(self) -> bool:
        return self.disposition is RepairControllerDisposition.ADMITTED

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REPAIR_CONTROLLER_RESULT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "disposition": self.disposition.value,
            "selected_tier": self.selected_tier.value,
            "reason_codes": list(self.reason_codes),
            "merge": self.merge.to_dict(),
            "plan_id": self.plan.plan_id,
            "envelope_id": self.envelope_id,
            "failure_signature": self.failure_signature,
            "mutation_class": self.mutation_class.value,
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "model_call_count": self.model_call_count,
            "source_edit_admitted": self.source_edit_admitted,
            "engine_invoked": self.engine_invoked,
            "materializer_invoked": self.materializer_invoked,
            "suffix_receipt_id": self.suffix_receipt_id,
            "authorizes_effect": False,
            "authorizes_merge": False,
        }
        if include_identity:
            payload["result_id"] = self.result_id
        return payload


def _bind_selected_plan(plan: AutonomousRepairPlan, selected_tier: RepairTier) -> AutonomousRepairPlan:
    if plan.repair_tier is selected_tier:
        return plan
    payload = plan.to_dict()
    payload["repair_tier"] = selected_tier.value
    payload.pop("plan_id", None)
    payload.pop("content_id", None)
    return AutonomousRepairPlan.from_dict(payload)


class AutonomousRepairController:
    """Tier-selection and scope facade over the existing repair engine.

    The controller never constructs a second engine.  An
    ``AutonomousRepairEngine`` may be injected; if absent, the facade still
    performs admission, backoff, and merge evaluation without minting one.
    Byte mutation is delegated to ``AutonomousRepairMaterializer``.
    """

    INTERFACE: ClassVar[str] = AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    def __init__(
        self,
        *,
        engine: AutonomousRepairEngine | None = None,
        materializer: AutonomousRepairMaterializer | None = None,
        model_invoker: Callable[[RepairControllerRequest], Mapping[str, Any]] | None = None,
        base_backoff_milliseconds: int = DEFAULT_BASE_BACKOFF_MS,
        max_backoff_milliseconds: int = DEFAULT_MAX_BACKOFF_MS,
        max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES,
    ) -> None:
        if engine is not None and not isinstance(engine, AutonomousRepairEngine):
            raise RepairControllerError("engine must be the existing AutonomousRepairEngine")
        if materializer is not None and not isinstance(materializer, AutonomousRepairMaterializer):
            raise RepairControllerError(
                "materializer must be the existing AutonomousRepairMaterializer"
            )
        if model_invoker is not None and not callable(model_invoker):
            raise RepairControllerError("model_invoker must be callable")
        self._engine = engine
        self._materializer = materializer
        self._model_invoker = model_invoker
        self._base_backoff_ms = _int(
            base_backoff_milliseconds, "base_backoff_milliseconds", minimum=1
        )
        self._max_backoff_ms = _int(
            max_backoff_milliseconds, "max_backoff_milliseconds", minimum=self._base_backoff_ms
        )
        self._max_identical = _int(
            max_identical_failures, "max_identical_failures", minimum=1
        )
        self._failures: dict[str, RepairFailureRecord] = {}
        self._lock = RLock()

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine(self) -> AutonomousRepairEngine | None:
        return self._engine

    @property
    def materializer(self) -> AutonomousRepairMaterializer | None:
        return self._materializer

    @property
    def creates_repair_engine(self) -> bool:
        return False

    def failure_record(self, signature: str) -> RepairFailureRecord | None:
        with self._lock:
            return self._failures.get(signature)

    def record_failure(
        self,
        request: RepairControllerRequest,
        *,
        diagnostic_receipt_id: str = "",
        model_calls: int = 0,
    ) -> RepairFailureRecord:
        """Record one identical-failure observation for backoff."""

        if not isinstance(request, RepairControllerRequest):
            raise RepairControllerError("request must be a RepairControllerRequest")
        signature = request.failure_signature
        diagnostic = diagnostic_receipt_id or request.diagnostic_receipt_id
        with self._lock:
            previous = self._failures.get(signature)
            count = 1 if previous is None else previous.count + 1
            exponent = max(0, count - 1)
            backoff = min(self._max_backoff_ms, self._base_backoff_ms * (2**exponent))
            record = RepairFailureRecord(
                signature=signature,
                diagnostic_receipt_id=diagnostic,
                count=count,
                backoff_milliseconds=backoff,
                model_calls=(0 if previous is None else previous.model_calls) + model_calls,
            )
            if len(self._failures) >= MAX_FAILURE_RECORDS and signature not in self._failures:
                oldest = next(iter(self._failures))
                self._failures.pop(oldest, None)
            self._failures[signature] = record
            return record

    def select_tier(self, request: RepairControllerRequest) -> RepairTier:
        if not isinstance(request, RepairControllerRequest):
            raise RepairControllerError("request must be a RepairControllerRequest")
        return select_repair_tier(
            requested=request.plan.repair_tier,
            deterministic_applicable=request.deterministic_applicable,
            template_applicable=request.template_applicable,
            model_assistance_requested=request.model_assistance_requested,
        )

    def _mutation_class(self, request: RepairControllerRequest) -> RepairMutationClass:
        allowed = tuple(request.envelope.allowed_paths) or tuple(request.plan.allowed_paths)
        paths = (*request.plan.predicted_files, *request.changed_paths)
        if request.source_edit_operator is not None:
            relative = request.source_edit_operator.get("relative_path")
            if isinstance(relative, str) and relative.strip():
                paths = (*paths, relative.strip().replace("\\", "/"))
        classes = [
            classify_mutation(
                path,
                allowed_paths=allowed,
                forbidden_symbols=request.plan.forbidden_symbols,
                symbols=request.plan.predicted_symbols,
            )
            for path in paths
        ]
        if any(item is RepairMutationClass.SELF_EDIT for item in classes):
            return RepairMutationClass.SELF_EDIT
        if any(item is RepairMutationClass.VALIDATOR_POLICY_KEY for item in classes):
            return RepairMutationClass.VALIDATOR_POLICY_KEY
        if any(item is RepairMutationClass.SCOPE_ESCAPE for item in classes):
            return RepairMutationClass.SCOPE_ESCAPE
        if not allowed:
            return RepairMutationClass.SCOPE_ESCAPE
        return RepairMutationClass.ADMITTED_SCOPE

    def _source_edit_admission(
        self, request: RepairControllerRequest
    ) -> tuple[bool, tuple[str, ...]]:
        operator = request.source_edit_operator
        if operator is None:
            if request.apply_source_edit:
                return False, ("typed_admitted_source_edit_operator_required",)
            return False, ("source_edit_not_requested",)
        try:
            parsed = AdmittedSourceEditOperator.from_mapping(operator)
        except AdmittedSourceEditError as exc:
            return False, (str(exc) or "source_edit_operator_not_admitted",)
        relative = parsed.relative_path.replace("\\", "/")
        if relative not in request.plan.predicted_files and not path_in_prefixes(
            relative, request.plan.predicted_files
        ):
            return False, ("source_edit_path_outside_predicted_files",)
        mutation = classify_mutation(
            relative,
            allowed_paths=tuple(request.envelope.allowed_paths) or tuple(request.plan.allowed_paths),
            forbidden_symbols=request.plan.forbidden_symbols,
            symbols=request.plan.predicted_symbols,
        )
        if mutation is not RepairMutationClass.ADMITTED_SCOPE:
            return False, (mutation.value,)
        return True, ("admitted_source_edit_operator",)

    def _model_assisted_requirements(
        self, request: RepairControllerRequest, selected: RepairTier
    ) -> tuple[str, ...]:
        if selected is not RepairTier.MODEL_ASSISTED_BOUNDED:
            return ()
        missing: list[str] = []
        if not request.plan.worktree_id or not request.isolated_worktree:
            missing.append("model_assisted_requires_isolated_worktree")
        if not request.plan.context_reference_ids:
            missing.append("model_assisted_requires_context_references")
        if not request.plan.required_test_ids and not request.envelope.required_test_ids:
            missing.append("model_assisted_requires_predetermined_tests")
        if not request.plan.predicted_files or not request.plan.predicted_symbols:
            missing.append("model_assisted_requires_exact_paths_and_symbols")
        return tuple(missing)

    def _merge_evaluation(
        self,
        request: RepairControllerRequest,
        *,
        selected: RepairTier,
        considered: bool,
        terminal_status: TerminalStatus,
    ) -> RepairMergeEvaluation:
        conditions = evaluate_merge_conjunction(
            policy=request.policy,
            envelope=request.envelope,
            plan=request.plan,
            selected_tier=selected,
            changed_paths=request.changed_paths,
            changed_line_count=request.changed_line_count,
            validation_receipt_ids=request.validation_receipt_ids,
            proof_receipt_ids=request.proof_receipt_ids,
            terminal_status=terminal_status,
            isolated_worktree=request.isolated_worktree,
        )
        return RepairMergeEvaluation.from_conditions(
            conditions,
            considered=considered,
            risk_class=request.envelope.risk_assessment.risk_class,
        )

    def _receipt(
        self,
        request: RepairControllerRequest,
        *,
        terminal_status: TerminalStatus,
        changed_paths: Sequence[str],
    ) -> AutonomousRepairReceipt:
        return AutonomousRepairReceipt(
            plan_id=request.plan.plan_id,
            envelope_id=request.envelope.envelope_id,
            terminal_status=terminal_status,
            changed_paths=tuple(changed_paths),
            validation_receipt_ids=request.validation_receipt_ids,
            proof_receipt_ids=request.proof_receipt_ids,
            adversarial_assurance_receipt_ids=request.adversarial_assurance_receipt_ids,
            rollback_receipt_id=request.plan.rollback_plan_id if request.roll_back else "",
            failure_signature=request.failure_signature
            if terminal_status is TerminalStatus.FAILED
            else "",
            diagnostic_receipt_id=request.diagnostic_receipt_id,
            authorizes_merge=False,
        )

    def _result(
        self,
        request: RepairControllerRequest,
        *,
        disposition: RepairControllerDisposition,
        selected: RepairTier,
        reason_codes: Sequence[str],
        mutation_class: RepairMutationClass,
        merge: RepairMergeEvaluation,
        receipt: AutonomousRepairReceipt | None = None,
        diagnostic_reused: bool = False,
        backoff_milliseconds: int = 0,
        model_call_count: int = 0,
        source_edit_admitted: bool = False,
        engine_invoked: bool = False,
        materializer_invoked: bool = False,
        plan: AutonomousRepairPlan | None = None,
    ) -> RepairControllerResult:
        return RepairControllerResult(
            disposition=disposition,
            selected_tier=selected,
            reason_codes=tuple(reason_codes),
            merge=merge,
            plan=plan or request.plan,
            envelope_id=request.envelope.envelope_id,
            failure_signature=request.failure_signature,
            mutation_class=mutation_class,
            receipt=receipt,
            diagnostic_reused=diagnostic_reused,
            backoff_milliseconds=backoff_milliseconds,
            model_call_count=model_call_count,
            source_edit_admitted=source_edit_admitted,
            engine_invoked=engine_invoked,
            materializer_invoked=materializer_invoked,
            suffix_receipt_id="" if request.suffix_receipt is None else request.suffix_receipt.receipt_id,
        )

    def repair(self, request: RepairControllerRequest) -> RepairControllerResult:
        """Admit one bounded repair, reuse identical-failure diagnosis, and stop."""

        if not isinstance(request, RepairControllerRequest):
            raise RepairControllerError("request must be a RepairControllerRequest")
        selected = self.select_tier(request)
        bound_plan = _bind_selected_plan(request.plan, selected)
        mutation = self._mutation_class(request)
        if request.roll_back:
            merge = self._merge_evaluation(
                request,
                selected=selected,
                considered=False,
                terminal_status=TerminalStatus.CANCELLED,
            )
            receipt = self._receipt(
                request, terminal_status=TerminalStatus.CANCELLED, changed_paths=()
            )
            return self._result(
                request,
                disposition=RepairControllerDisposition.ROLLED_BACK,
                selected=selected,
                reason_codes=("rollback_plan_invoked", "isolated_worktree_discarded"),
                mutation_class=mutation,
                merge=merge,
                receipt=receipt,
                plan=bound_plan,
            )

        if mutation is RepairMutationClass.SCOPE_ESCAPE:
            merge = self._merge_evaluation(
                request, selected=selected, considered=False, terminal_status=TerminalStatus.FAILED
            )
            return self._result(
                request,
                disposition=RepairControllerDisposition.REJECTED,
                selected=selected,
                reason_codes=("scope_escape", "admitted_envelope_required"),
                mutation_class=mutation,
                merge=merge,
                plan=bound_plan,
            )
        if mutation is RepairMutationClass.SELF_EDIT:
            merge = self._merge_evaluation(
                request, selected=selected, considered=False, terminal_status=TerminalStatus.FAILED
            )
            return self._result(
                request,
                disposition=RepairControllerDisposition.REJECTED,
                selected=selected,
                reason_codes=("self_edit", "self_protecting_authority_path"),
                mutation_class=mutation,
                merge=merge,
                plan=bound_plan,
            )
        if mutation is RepairMutationClass.VALIDATOR_POLICY_KEY:
            merge = self._merge_evaluation(
                request, selected=selected, considered=False, terminal_status=TerminalStatus.FAILED
            )
            return self._result(
                request,
                disposition=RepairControllerDisposition.REJECTED,
                selected=selected,
                reason_codes=("validator_policy_key_mutation", "protected_authority_path"),
                mutation_class=mutation,
                merge=merge,
                plan=bound_plan,
            )

        source_ok, source_reasons = self._source_edit_admission(request)
        policy_flag_set = self._engine is not None and bool(
            getattr(self._engine.policy, "allow_code_edit_materialize", False)
        )
        if request.apply_source_edit and not source_ok:
            reasons = list(source_reasons or ("source_edit_operator_not_admitted",))
            if policy_flag_set and "policy_flag_is_not_source_edit_admission" not in reasons:
                reasons.insert(0, "policy_flag_is_not_source_edit_admission")
            merge = self._merge_evaluation(
                request, selected=selected, considered=False, terminal_status=TerminalStatus.FAILED
            )
            return self._result(
                request,
                disposition=RepairControllerDisposition.REJECTED,
                selected=selected,
                reason_codes=tuple(reasons),
                mutation_class=mutation,
                merge=merge,
                plan=bound_plan,
            )

        missing = self._model_assisted_requirements(request, selected)
        if missing:
            merge = self._merge_evaluation(
                request, selected=selected, considered=False, terminal_status=TerminalStatus.FAILED
            )
            return self._result(
                request,
                disposition=RepairControllerDisposition.REJECTED,
                selected=selected,
                reason_codes=missing,
                mutation_class=mutation,
                merge=merge,
                plan=bound_plan,
            )

        with self._lock:
            previous = self._failures.get(request.failure_signature)
        if previous is not None:
            if previous.count >= self._max_identical:
                merge = self._merge_evaluation(
                    request,
                    selected=selected,
                    considered=False,
                    terminal_status=TerminalStatus.EXHAUSTED,
                )
                return self._result(
                    request,
                    disposition=RepairControllerDisposition.EXHAUSTED,
                    selected=selected,
                    reason_codes=("identical_failure_exhausted", "diagnosis_reused"),
                    mutation_class=mutation,
                    merge=merge,
                    diagnostic_reused=True,
                    backoff_milliseconds=previous.backoff_milliseconds,
                    model_call_count=0,
                    source_edit_admitted=source_ok,
                    plan=bound_plan,
                )
            # Count this retry as another identical observation without a model call.
            prior_backoff = previous.backoff_milliseconds
            self.record_failure(request, diagnostic_receipt_id=previous.diagnostic_receipt_id)
            merge = self._merge_evaluation(
                request,
                selected=selected,
                considered=False,
                terminal_status=TerminalStatus.BLOCKED,
            )
            return self._result(
                request,
                disposition=RepairControllerDisposition.BACKED_OFF,
                selected=selected,
                reason_codes=("identical_failure_backoff", "diagnosis_reused", "no_repeated_model_call"),
                mutation_class=mutation,
                merge=merge,
                diagnostic_reused=True,
                backoff_milliseconds=prior_backoff,
                model_call_count=0,
                source_edit_admitted=source_ok,
                plan=bound_plan,
            )

        model_calls = 0
        if selected is RepairTier.MODEL_ASSISTED_BOUNDED and self._model_invoker is not None:
            try:
                invoked = self._model_invoker(request)
            except Exception:
                self.record_failure(request, model_calls=1)
                merge = self._merge_evaluation(
                    request,
                    selected=selected,
                    considered=False,
                    terminal_status=TerminalStatus.FAILED,
                )
                return self._result(
                    request,
                    disposition=RepairControllerDisposition.REJECTED,
                    selected=selected,
                    reason_codes=("model_assisted_failed",),
                    mutation_class=mutation,
                    merge=merge,
                    model_call_count=1,
                    source_edit_admitted=source_ok,
                    plan=bound_plan,
                )
            model_calls = 1
            if isinstance(invoked, Mapping) and invoked.get("failed") is True:
                self.record_failure(request, model_calls=1)
                merge = self._merge_evaluation(
                    request,
                    selected=selected,
                    considered=False,
                    terminal_status=TerminalStatus.FAILED,
                )
                return self._result(
                    request,
                    disposition=RepairControllerDisposition.REJECTED,
                    selected=selected,
                    reason_codes=("model_assisted_failed", "failure_recorded"),
                    mutation_class=mutation,
                    merge=merge,
                    model_call_count=1,
                    source_edit_admitted=source_ok,
                    plan=bound_plan,
                )

        engine_invoked = False
        if request.delegate_to_engine and self._engine is None:
            merge = self._merge_evaluation(
                request,
                selected=selected,
                considered=False,
                terminal_status=TerminalStatus.UNAVAILABLE,
            )
            return self._result(
                request,
                disposition=RepairControllerDisposition.REJECTED,
                selected=selected,
                reason_codes=("existing_engine_required", "no_second_repair_engine"),
                mutation_class=mutation,
                merge=merge,
                model_call_count=model_calls,
                source_edit_admitted=source_ok,
                plan=bound_plan,
            )
        if request.delegate_to_engine and self._engine is not None:
            from ..autonomous_repair.contracts import RepairWorkItem

            work = RepairWorkItem(
                work_id=request.plan.plan_id,
                operation=request.plan.predicted_symbols[0] if request.plan.predicted_symbols else "repair",
                path=request.plan.predicted_files[0],
                symbol=request.plan.predicted_symbols[0] if request.plan.predicted_symbols else "",
                write_paths=request.plan.predicted_files,
                domain="agent_supervisor",
            )
            report = self._engine.run((work,))
            engine_invoked = True
            if int(getattr(report, "model_call_count", 0) or 0) != 0:
                raise RepairControllerError("existing repair engine must remain no-LLM")

        materializer_invoked = False
        if request.apply_source_edit and source_ok:
            if self._materializer is None:
                merge = self._merge_evaluation(
                    request,
                    selected=selected,
                    considered=False,
                    terminal_status=TerminalStatus.FAILED,
                )
                return self._result(
                    request,
                    disposition=RepairControllerDisposition.REJECTED,
                    selected=selected,
                    reason_codes=("materializer_required_for_source_edit", "no_second_repair_engine"),
                    mutation_class=mutation,
                    merge=merge,
                    model_call_count=model_calls,
                    source_edit_admitted=source_ok,
                    engine_invoked=engine_invoked,
                    plan=bound_plan,
                )
            preferred = (
                str(request.source_edit_operator.get("relative_path") or "")
                if request.source_edit_operator
                else request.plan.predicted_files[0]
            )
            plan_payload = {
                "plan_id": request.plan.plan_id,
                "work_id": request.plan.plan_id,
                "operation": (
                    request.plan.predicted_symbols[0] if request.plan.predicted_symbols else "repair"
                ),
                "materialize_ready": True,
                "preferred_path": preferred,
                "handler": request.plan.predicted_symbols[0] if request.plan.predicted_symbols else "",
                "source_edit_operator": dict(request.source_edit_operator or {}),
            }
            try:
                self._materializer.materialize_plans([plan_payload])
            except Exception:
                merge = self._merge_evaluation(
                    request,
                    selected=selected,
                    considered=False,
                    terminal_status=TerminalStatus.FAILED,
                )
                return self._result(
                    request,
                    disposition=RepairControllerDisposition.REJECTED,
                    selected=selected,
                    reason_codes=("materializer_rejected",),
                    mutation_class=mutation,
                    merge=merge,
                    model_call_count=model_calls,
                    source_edit_admitted=source_ok,
                    engine_invoked=engine_invoked,
                    plan=bound_plan,
                )
            materializer_invoked = True

        terminal = TerminalStatus.SUCCEEDED
        if not request.validation_receipt_ids:
            terminal = TerminalStatus.PENDING
        merge = self._merge_evaluation(
            request,
            selected=selected,
            considered=terminal is TerminalStatus.SUCCEEDED,
            terminal_status=terminal if terminal is TerminalStatus.SUCCEEDED else TerminalStatus.PENDING,
        )
        receipt = None
        if terminal is TerminalStatus.SUCCEEDED:
            receipt = self._receipt(
                request,
                terminal_status=terminal,
                changed_paths=request.changed_paths or request.plan.predicted_files,
            )
        reasons = ["envelope_bound", f"tier:{selected.value}"]
        if source_ok:
            reasons.append("source_edit_operator_admitted")
        if request.suffix_receipt is not None:
            reasons.append("suffix_contract_bound")
        if merge.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE:
            reasons.append("r2_merge_conjunction_satisfied")
        elif merge.disposition is RepairMergeDisposition.PROPOSAL_ONLY:
            reasons.append("r3_or_incomplete_conjunction_is_proposal")
        return self._result(
            request,
            disposition=RepairControllerDisposition.ADMITTED,
            selected=selected,
            reason_codes=reasons,
            mutation_class=mutation,
            merge=merge,
            receipt=receipt,
            model_call_count=model_calls,
            source_edit_admitted=source_ok,
            engine_invoked=engine_invoked,
            materializer_invoked=materializer_invoked,
            plan=bound_plan,
        )
