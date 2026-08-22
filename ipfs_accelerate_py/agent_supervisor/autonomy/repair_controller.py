# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded facade over the existing autonomous-repair engine.

``AutonomousRepairController@1`` selects among deterministic, template-
constrained, and model-assisted tiers, then binds an exact envelope, isolated
worktree, predetermined checks, identical-failure backoff, and merge
disposition.  It is not a second repair engine: admission and execution stay
with ``agent_supervisor.autonomous_repair`` (and ``DecisionRuntime`` for any
later effect).  Repair receipts remain evidence and never authorize merge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..autonomous_repair.engine import AutonomousRepairEngine
from ..autonomous_repair.materialize import (
    AdmittedSourceEditError,
    AdmittedSourceEditOperator,
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

AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE: Final[str] = "AutonomousRepairController@1"
AUTONOMOUS_REPAIR_CONTROLLER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller@1"
)
REPAIR_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-admission@1"
)
REPAIR_CONTROLLER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-result@1"
)
REPAIR_MERGE_ELIGIBILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-merge-eligibility@1"
)
REPAIR_FAILURE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-failure-record@1"
)

CONTROLLER_RELATIVE_PATH: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
)
ENGINE_PACKAGE_PREFIX: Final[str] = "ipfs_accelerate_py/agent_supervisor/autonomous_repair"

MAX_REPAIR_ATTEMPT_BYTES: Final[int] = 4 * MAX_CANONICAL_RECORD_BYTES
MAX_FAILURE_RECORDS: Final[int] = 1_024
DEFAULT_BACKOFF_MS: Final[int] = 100
MAX_BACKOFF_MS: Final[int] = 10_000

_LOW_RISK_CLASSES: Final[frozenset[RiskClass]] = frozenset(
    {
        RiskClass.R0_PURE,
        RiskClass.R1_READ_ONLY,
        RiskClass.R2_REVERSIBLE_LOCAL,
    }
)
_EXECUTABLE_LEVELS: Final[frozenset[AutonomyLevel]] = frozenset(
    {
        AutonomyLevel.EXECUTE_REVERSIBLE,
        AutonomyLevel.EXECUTE_BOUNDED_MUTATION,
        AutonomyLevel.SELF_REPAIR_ISOLATED,
    }
)

# Every stated low-risk autonomous-merge conjunction.  R3+ is proposal-only
# even when these hold.  The conjunction never grants merge authority.
LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "policy_autonomous_merge_enabled",
    "risk_class_r2_or_lower",
    "envelope_reversible",
    "risk_assessment_reversible",
    "autonomy_level_allows_execution",
    "exact_paths_bound",
    "exact_symbols_bound",
    "bounded_patch_envelope",
    "isolated_worktree_bound",
    "predetermined_tests_present",
    "predetermined_proofs_satisfied",
    "validation_receipts_current",
    "rollback_plan_bound",
    "no_scope_escape",
    "no_self_edit",
    "no_validator_policy_key_mutation",
    "no_protected_authority_path_mutation",
    "repair_succeeded",
    "changed_paths_within_envelope",
)

SELF_EDIT_PATHS: Final[frozenset[str]] = frozenset(
    {
        CONTROLLER_RELATIVE_PATH,
        "test/api/autonomy/test_repair_controller.py",
        "test/api/test_agent_supervisor_autonomous_repair_source_edit_admission.py",
    }
)

# Operator-protected authority surfaces plus repair-engine authority.  A repair
# may not edit these even when an envelope lists them.
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
        CONTROLLER_RELATIVE_PATH,
    }
)

PROTECTED_AUTHORITY_PREFIXES: Final[tuple[str, ...]] = (
    ENGINE_PACKAGE_PREFIX,
    "ipfs_accelerate_py/agent_supervisor/validation/",
    "ipfs_accelerate_py/agent_supervisor/proof/",
    "ipfs_accelerate_py/agent_supervisor/merge/",
)

_VALIDATOR_POLICY_KEY_PATHS: Final[frozenset[str]] = frozenset(
    {
        "ipfs_accelerate_py/llm_router.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "config/agent_supervisor_autonomous_meta_controller_scheduler.json",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py",
    }
)
_VALIDATOR_POLICY_KEY_PREFIXES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py/agent_supervisor/validation/",
    "config/",
)
_VALIDATOR_POLICY_KEY_BASENAME_MARKERS: Final[tuple[str, ...]] = (
    "authority_policy",
    "trusted_key",
    "trusted-key",
    "validator_policy",
    "promotion_key",
)
_VALIDATOR_POLICY_KEY_SUFFIXES: Final[tuple[str, ...]] = (
    ".pem",
    ".key",
    ".pub",
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
    ADMITTED = "admitted"
    EXECUTED = "executed"
    REJECTED = "rejected"
    BACKOFF = "backoff"
    ROLLBACK = "rollback"
    PROPOSAL = "proposal"
    AUTONOMOUS_MERGE = "autonomous_merge"


class RepairMergeDisposition(str, Enum):
    AUTONOMOUS_MERGE = "autonomous_merge"
    PROPOSAL = "proposal"
    REJECTED = "rejected"
    BACKOFF = "backoff"
    INELIGIBLE = "ineligible"


class RepairPathViolation(str, Enum):
    NONE = "none"
    SCOPE_ESCAPE = "scope_escape"
    SELF_EDIT = "self_edit"
    VALIDATOR_POLICY_KEY = "validator_policy_key"
    PROTECTED_AUTHORITY = "protected_authority"


def _posix_path(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    result = value.strip().replace("\\", "/")
    parsed = PurePosixPath(result)
    if (
        not result
        or parsed.is_absolute()
        or ".." in parsed.parts
        or "\x00" in result
        or result in {".", ""}
    ):
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    return parsed.as_posix()


def _path_in_prefixes(path: str, prefixes: Sequence[str]) -> bool:
    for prefix in prefixes:
        if not prefix:
            continue
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return True
    return False


def normalize_repair_path(value: Any, name: str = "path") -> str:
    """Normalize one repository-relative POSIX path used by admission."""

    return _posix_path(value, name)


def is_self_edit_path(path: str) -> bool:
    relative = _posix_path(path, "path")
    return relative in SELF_EDIT_PATHS or relative == CONTROLLER_RELATIVE_PATH


def is_validator_policy_key_path(path: str) -> bool:
    relative = _posix_path(path, "path")
    if relative in _VALIDATOR_POLICY_KEY_PATHS:
        return True
    if _path_in_prefixes(relative, _VALIDATOR_POLICY_KEY_PREFIXES):
        basename = PurePosixPath(relative).name.lower()
        if any(
            marker in basename or marker in relative.lower()
            for marker in _VALIDATOR_POLICY_KEY_BASENAME_MARKERS
        ):
            return True
        if relative.startswith("config/") and relative.endswith(".json"):
            return True
        if relative.startswith("ipfs_accelerate_py/agent_supervisor/validation/"):
            return True
    basename = PurePosixPath(relative).name.lower()
    if any(basename.endswith(suffix) for suffix in _VALIDATOR_POLICY_KEY_SUFFIXES):
        return True
    if any(marker in basename for marker in _VALIDATOR_POLICY_KEY_BASENAME_MARKERS):
        return True
    return False


def is_protected_authority_path(path: str) -> bool:
    relative = _posix_path(path, "path")
    if relative in PROTECTED_AUTHORITY_PATHS or relative in SELF_EDIT_PATHS:
        return True
    return _path_in_prefixes(relative, PROTECTED_AUTHORITY_PREFIXES)


def classify_repair_path(path: str, *, allowed_paths: Sequence[str]) -> RepairPathViolation:
    """Classify one proposed mutation path against envelope and authority gates."""

    relative = _posix_path(path, "path")
    if is_self_edit_path(relative):
        return RepairPathViolation.SELF_EDIT
    if is_validator_policy_key_path(relative):
        return RepairPathViolation.VALIDATOR_POLICY_KEY
    if is_protected_authority_path(relative):
        return RepairPathViolation.PROTECTED_AUTHORITY
    if allowed_paths and not _path_in_prefixes(relative, allowed_paths):
        return RepairPathViolation.SCOPE_ESCAPE
    return RepairPathViolation.NONE


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


def _paths(
    value: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_SEQUENCE_ITEMS,
) -> tuple[str, ...]:
    if value is None:
        raw: Sequence[Any] = ()
    elif isinstance(value, str):
        raw = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise RepairControllerError(f"{name} must be a sequence of paths")
    if len(raw) > maximum:
        raise RepairControllerError(f"{name} contains too many items")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        path = _posix_path(item, name)
        if path not in seen:
            seen.add(path)
            normalized.append(path)
    if required and not normalized:
        raise RepairControllerError(f"{name} must not be empty")
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


def _backoff_ms(count: int) -> int:
    if count <= 0:
        return 0
    shift = min(count - 1, 16)
    value = DEFAULT_BACKOFF_MS * (1 << shift)
    return min(value, MAX_BACKOFF_MS)


def _reason_code(violation: RepairPathViolation) -> str:
    if violation is RepairPathViolation.SCOPE_ESCAPE:
        return "scope_escape"
    if violation is RepairPathViolation.SELF_EDIT:
        return "self_edit"
    if violation is RepairPathViolation.VALIDATOR_POLICY_KEY:
        return "validator_policy_key_mutation"
    if violation is RepairPathViolation.PROTECTED_AUTHORITY:
        return "protected_authority_path"
    return "admitted"


def classify_paths(
    paths: Sequence[str],
    *,
    allowed_paths: Sequence[str],
) -> tuple[RepairPathViolation, tuple[str, ...]]:
    worst = RepairPathViolation.NONE
    reasons: list[str] = []
    order = (
        RepairPathViolation.SELF_EDIT,
        RepairPathViolation.VALIDATOR_POLICY_KEY,
        RepairPathViolation.PROTECTED_AUTHORITY,
        RepairPathViolation.SCOPE_ESCAPE,
    )
    rank = {item: index for index, item in enumerate(order)}
    for path in paths:
        violation = classify_repair_path(path, allowed_paths=allowed_paths)
        if violation is RepairPathViolation.NONE:
            continue
        code = _reason_code(violation)
        if code not in reasons:
            reasons.append(code)
        if worst is RepairPathViolation.NONE or rank[violation] < rank.get(worst, 99):
            worst = violation
    return worst, tuple(reasons)


@dataclass(frozen=True)
class RepairAttempt:
    """One envelope-bound repair request presented to the facade."""

    envelope: AutonomyEnvelope
    policy: AutonomyPolicy
    plan: AutonomousRepairPlan
    work_items: tuple[Any, ...] = ()
    changed_paths: tuple[str, ...] = ()
    changed_symbols: tuple[str, ...] = ()
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    source_edit_operator: Mapping[str, Any] | AdmittedSourceEditOperator | None = None
    isolated_worktree_root: str = ""
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""
    new_evidence_since_failure: bool = False
    apply_source_edit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, AutonomyEnvelope):
            raise RepairControllerError("envelope must be AutonomyEnvelope")
        if not isinstance(self.policy, AutonomyPolicy):
            raise RepairControllerError("policy must be AutonomyPolicy")
        if not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be AutonomousRepairPlan")
        if self.envelope.policy_id != self.policy.policy_id:
            raise RepairControllerError("envelope policy_id does not match policy")
        if self.plan.patch_envelope_id != self.envelope.envelope_id:
            raise RepairControllerError("repair plan is not bound to the envelope")
        if self.plan.objective_id != self.envelope.objective_id:
            raise RepairControllerError("repair plan objective does not match envelope")
        if self.plan.task_id != self.envelope.task_id:
            raise RepairControllerError("repair plan task does not match envelope")
        if len(self.work_items) > MAX_SEQUENCE_ITEMS:
            raise RepairControllerError("work_items exceeds the bounded collection size")
        object.__setattr__(self, "changed_paths", _paths(self.changed_paths, "changed_paths"))
        object.__setattr__(
            self, "changed_symbols", _identifiers(self.changed_symbols, "changed_symbols")
        )
        for name in (
            "validation_receipt_ids",
            "proof_receipt_ids",
            "adversarial_assurance_receipt_ids",
        ):
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
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
            "new_evidence_since_failure",
            _bool(self.new_evidence_since_failure, "new_evidence_since_failure"),
        )
        object.__setattr__(
            self, "apply_source_edit", _bool(self.apply_source_edit, "apply_source_edit")
        )
        if self.isolated_worktree_root:
            root = str(self.isolated_worktree_root).strip()
            if not root or "\x00" in root:
                raise RepairControllerError("isolated_worktree_root is unsafe")
            object.__setattr__(self, "isolated_worktree_root", root)
        if isinstance(self.source_edit_operator, Mapping) and not isinstance(
            self.source_edit_operator, AdmittedSourceEditOperator
        ):
            _reject_forbidden_keys(self.source_edit_operator, "source_edit_operator")
            object.__setattr__(
                self, "source_edit_operator", MappingProxyType(dict(self.source_edit_operator))
            )

    @property
    def proposed_paths(self) -> tuple[str, ...]:
        paths = list(self.plan.predicted_files)
        paths.extend(self.changed_paths)
        if isinstance(self.source_edit_operator, AdmittedSourceEditOperator):
            paths.append(self.source_edit_operator.relative_path)
        elif isinstance(self.source_edit_operator, Mapping):
            relative = self.source_edit_operator.get("relative_path")
            if isinstance(relative, str) and relative.strip():
                paths.append(_posix_path(relative, "relative_path"))
        unique = tuple(sorted(dict.fromkeys(paths)))
        return unique

    @property
    def proposed_symbols(self) -> tuple[str, ...]:
        symbols = list(self.plan.predicted_symbols)
        symbols.extend(self.changed_symbols)
        return tuple(sorted(dict.fromkeys(symbols)))


@dataclass(frozen=True)
class RepairAdmission:
    """Fail-closed admission of one repair attempt."""

    admitted: bool
    violation: RepairPathViolation
    reason_codes: tuple[str, ...]
    selected_tier: RepairTier
    proposed_paths: tuple[str, ...]
    proposed_symbols: tuple[str, ...]
    source_edit: AdmittedSourceEditOperator | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(
            self, "violation", _enum(self.violation, RepairPathViolation, "violation")
        )
        object.__setattr__(
            self, "reason_codes", _identifiers(self.reason_codes, "reason_codes", preserve_order=True)
        )
        object.__setattr__(
            self, "selected_tier", _enum(self.selected_tier, RepairTier, "selected_tier")
        )
        object.__setattr__(self, "proposed_paths", _paths(self.proposed_paths, "proposed_paths"))
        object.__setattr__(
            self, "proposed_symbols", _identifiers(self.proposed_symbols, "proposed_symbols")
        )
        if self.source_edit is not None and not isinstance(
            self.source_edit, AdmittedSourceEditOperator
        ):
            raise RepairControllerError("source_edit must be AdmittedSourceEditOperator")
        if self.admitted and self.violation is not RepairPathViolation.NONE:
            raise RepairControllerError("an admitted repair cannot carry a path violation")
        if not self.admitted and not self.reason_codes:
            raise RepairControllerError("a rejected admission requires reason codes")

    def to_dict(self) -> Mapping[str, Any]:
        payload = {
            "schema": REPAIR_ADMISSION_SCHEMA,
            "admitted": self.admitted,
            "violation": self.violation.value,
            "reason_codes": list(self.reason_codes),
            "selected_tier": self.selected_tier.value,
            "proposed_paths": list(self.proposed_paths),
            "proposed_symbols": list(self.proposed_symbols),
            "source_edit_operator_id": "" if self.source_edit is None else self.source_edit.operator_id,
        }
        return MappingProxyType(payload)


@dataclass(frozen=True)
class RepairMergeEligibility:
    """Whether every stated low-risk merge condition holds.

    ``authorizes_merge`` is always false.  Eligibility is evidence for the
    existing merge authority; this facade cannot complete a merge.
    """

    eligible: bool
    disposition: RepairMergeDisposition
    satisfied_conditions: tuple[str, ...]
    missing_conditions: tuple[str, ...]
    authorizes_merge: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "eligible", _bool(self.eligible, "eligible"))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, RepairMergeDisposition, "disposition")
        )
        object.__setattr__(
            self,
            "satisfied_conditions",
            _identifiers(self.satisfied_conditions, "satisfied_conditions", preserve_order=True),
        )
        object.__setattr__(
            self,
            "missing_conditions",
            _identifiers(self.missing_conditions, "missing_conditions", preserve_order=True),
        )
        object.__setattr__(
            self, "authorizes_merge", _bool(self.authorizes_merge, "authorizes_merge")
        )
        if self.authorizes_merge:
            raise RepairControllerError("repair controller cannot independently authorize merge")
        if set(self.satisfied_conditions).intersection(self.missing_conditions):
            raise RepairControllerError("merge conditions cannot be both satisfied and missing")
        expected = set(LOW_RISK_MERGE_CONDITIONS)
        reported = set(self.satisfied_conditions).union(self.missing_conditions)
        if reported != expected:
            raise RepairControllerError("merge eligibility must report every stated low-risk condition")
        if self.eligible and self.missing_conditions:
            raise RepairControllerError("eligible merge cannot omit a stated low-risk condition")
        if self.eligible and self.disposition is not RepairMergeDisposition.AUTONOMOUS_MERGE:
            raise RepairControllerError("eligible merge must use autonomous_merge disposition")
        if (
            not self.eligible
            and self.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE
        ):
            raise RepairControllerError("ineligible merge cannot claim autonomous_merge")

    def to_dict(self) -> Mapping[str, Any]:
        payload = {
            "schema": REPAIR_MERGE_ELIGIBILITY_SCHEMA,
            "eligible": self.eligible,
            "disposition": self.disposition.value,
            "satisfied_conditions": list(self.satisfied_conditions),
            "missing_conditions": list(self.missing_conditions),
            "authorizes_merge": False,
        }
        return MappingProxyType(payload)


@dataclass(frozen=True)
class RepairControllerResult:
    """Closed outcome of one controller invocation."""

    disposition: RepairControllerDisposition
    admission: RepairAdmission
    merge: RepairMergeEligibility
    receipt: AutonomousRepairReceipt
    selected_tier: RepairTier
    reason_codes: tuple[str, ...]
    model_call_count: int = 0
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    engine_invoked: bool = False
    engine_report: Mapping[str, Any] | None = None
    authorizes_effect: bool = False
    authorizes_merge: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairControllerDisposition, "disposition"),
        )
        if not isinstance(self.admission, RepairAdmission):
            raise RepairControllerError("admission must be RepairAdmission")
        if not isinstance(self.merge, RepairMergeEligibility):
            raise RepairControllerError("merge must be RepairMergeEligibility")
        if not isinstance(self.receipt, AutonomousRepairReceipt):
            raise RepairControllerError("receipt must be AutonomousRepairReceipt")
        object.__setattr__(
            self, "selected_tier", _enum(self.selected_tier, RepairTier, "selected_tier")
        )
        object.__setattr__(
            self, "reason_codes", _identifiers(self.reason_codes, "reason_codes", preserve_order=True)
        )
        object.__setattr__(
            self, "model_call_count", _int(self.model_call_count, "model_call_count")
        )
        object.__setattr__(
            self, "diagnostic_reused", _bool(self.diagnostic_reused, "diagnostic_reused")
        )
        object.__setattr__(
            self,
            "backoff_milliseconds",
            _int(self.backoff_milliseconds, "backoff_milliseconds"),
        )
        object.__setattr__(self, "engine_invoked", _bool(self.engine_invoked, "engine_invoked"))
        object.__setattr__(
            self, "authorizes_effect", _bool(self.authorizes_effect, "authorizes_effect")
        )
        object.__setattr__(
            self, "authorizes_merge", _bool(self.authorizes_merge, "authorizes_merge")
        )
        if self.authorizes_effect or self.authorizes_merge:
            raise RepairControllerError("repair controller cannot authorize effects or merge")
        if self.receipt.authorizes_merge:
            raise RepairControllerError("repair receipts cannot independently authorize merge")
        if self.engine_report is not None:
            if not isinstance(self.engine_report, Mapping):
                raise RepairControllerError("engine_report must be a mapping")
            if len(self.engine_report) > MAX_MAPPING_ITEMS:
                raise RepairControllerError("engine_report exceeds bounded size")
            object.__setattr__(self, "engine_report", MappingProxyType(dict(self.engine_report)))

    @property
    def result_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REPAIR_CONTROLLER_RESULT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "disposition": self.disposition.value,
            "admission": dict(self.admission.to_dict()),
            "merge": dict(self.merge.to_dict()),
            "receipt": self.receipt.to_dict(),
            "selected_tier": self.selected_tier.value,
            "reason_codes": list(self.reason_codes),
            "model_call_count": self.model_call_count,
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "engine_invoked": self.engine_invoked,
            "engine_report": None if self.engine_report is None else dict(self.engine_report),
            "authorizes_effect": False,
            "authorizes_merge": False,
        }
        encoded = canonical_json(payload)
        if len(encoded) > MAX_REPAIR_ATTEMPT_BYTES:
            raise RepairControllerError("repair controller result exceeds bounded size")
        if include_identity:
            payload["result_id"] = content_identity(payload)
        return payload


def _engine_report_payload(report: Any) -> Mapping[str, Any] | None:
    if report is None:
        return None
    if hasattr(report, "to_dict"):
        payload = report.to_dict()
        if isinstance(payload, Mapping):
            return {
                "interface": payload.get("interface"),
                "passed": payload.get("passed"),
                "model_call_count": payload.get("model_call_count"),
                "llm_used": payload.get("llm_used"),
                "summary": payload.get("summary"),
            }
    return None


def evaluate_merge_eligibility(
    attempt: RepairAttempt,
    *,
    admission: RepairAdmission,
    terminal_status: TerminalStatus,
    changed_paths: Sequence[str],
    validation_receipt_ids: Sequence[str],
    proof_receipt_ids: Sequence[str],
    backoff: bool = False,
) -> RepairMergeEligibility:
    """Score every stated low-risk merge condition without granting authority."""

    envelope = attempt.envelope
    policy = attempt.policy
    plan = attempt.plan
    paths = tuple(changed_paths) or attempt.proposed_paths
    violation, _codes = classify_paths(paths, allowed_paths=envelope.allowed_paths)
    checks: dict[str, bool] = {
        "policy_autonomous_merge_enabled": policy.autonomous_merge_enabled is True,
        "risk_class_r2_or_lower": (
            plan.risk_class in _LOW_RISK_CLASSES
            and envelope.risk_assessment.risk_class in _LOW_RISK_CLASSES
        ),
        "envelope_reversible": envelope.reversible is True,
        "risk_assessment_reversible": envelope.risk_assessment.reversible is True,
        "autonomy_level_allows_execution": envelope.autonomy_level in _EXECUTABLE_LEVELS
        and policy.allows(envelope.autonomy_level, envelope.risk_assessment.risk_class),
        "exact_paths_bound": bool(plan.predicted_files) and bool(envelope.allowed_paths),
        "exact_symbols_bound": bool(plan.predicted_symbols),
        "bounded_patch_envelope": plan.max_changed_files > 0
        and plan.max_changed_lines > 0
        and len(paths) <= plan.max_changed_files,
        "isolated_worktree_bound": bool(plan.worktree_id),
        "predetermined_tests_present": set(envelope.required_test_ids).issubset(
            plan.required_test_ids
        )
        and (not envelope.required_test_ids or bool(plan.required_test_ids)),
        "predetermined_proofs_satisfied": set(envelope.required_proof_ids).issubset(
            plan.required_proof_ids
        )
        and (not envelope.required_proof_ids or bool(proof_receipt_ids)),
        "validation_receipts_current": bool(validation_receipt_ids),
        "rollback_plan_bound": bool(plan.rollback_plan_id),
        "no_scope_escape": violation is not RepairPathViolation.SCOPE_ESCAPE
        and all(_path_in_prefixes(path, envelope.allowed_paths) for path in paths),
        "no_self_edit": violation is not RepairPathViolation.SELF_EDIT
        and not any(is_self_edit_path(path) for path in paths),
        "no_validator_policy_key_mutation": violation is not RepairPathViolation.VALIDATOR_POLICY_KEY
        and not any(is_validator_policy_key_path(path) for path in paths),
        "no_protected_authority_path_mutation": (
            violation is not RepairPathViolation.PROTECTED_AUTHORITY
            and not any(is_protected_authority_path(path) for path in paths)
        ),
        "repair_succeeded": terminal_status is TerminalStatus.SUCCEEDED
        and admission.admitted
        and not backoff,
        "changed_paths_within_envelope": all(
            _path_in_prefixes(path, plan.allowed_paths) and path in plan.predicted_files
            for path in paths
        )
        and all(_path_in_prefixes(path, envelope.allowed_paths) for path in paths),
    }
    satisfied = tuple(name for name in LOW_RISK_MERGE_CONDITIONS if checks[name])
    missing = tuple(name for name in LOW_RISK_MERGE_CONDITIONS if not checks[name])
    r3_or_higher = envelope.risk_assessment.risk_class.rank >= RiskClass.R3_BOUNDED_REPOSITORY_MUTATION.rank
    if backoff:
        disposition = RepairMergeDisposition.BACKOFF
        eligible = False
    elif not admission.admitted or terminal_status in {
        TerminalStatus.FAILED,
        TerminalStatus.BLOCKED,
        TerminalStatus.CANCELLED,
    }:
        disposition = RepairMergeDisposition.REJECTED
        eligible = False
    elif not missing and not r3_or_higher:
        disposition = RepairMergeDisposition.AUTONOMOUS_MERGE
        eligible = True
    elif r3_or_higher and "risk_class_r2_or_lower" in missing and set(missing) == {
        "risk_class_r2_or_lower"
    }:
        disposition = RepairMergeDisposition.PROPOSAL
        eligible = False
    elif r3_or_higher and admission.admitted and terminal_status is TerminalStatus.SUCCEEDED:
        disposition = RepairMergeDisposition.PROPOSAL
        eligible = False
    else:
        disposition = RepairMergeDisposition.INELIGIBLE
        eligible = False
    return RepairMergeEligibility(
        eligible=eligible,
        disposition=disposition,
        satisfied_conditions=satisfied,
        missing_conditions=missing,
        authorizes_merge=False,
    )


def _source_edit_mapping(operator: AdmittedSourceEditOperator) -> dict[str, Any]:
    return {
        "operator_id": operator.operator_id,
        "owner_root": operator.owner_root,
        "relative_path": operator.relative_path,
        "old_digest": operator.old_digest,
        "new_digest": operator.new_digest,
        "old_bytes_b64": operator.old_bytes_b64,
        "new_bytes_b64": operator.new_bytes_b64,
        "forward_diff": operator.forward_diff,
        "inverse_diff": operator.inverse_diff,
        "disposition": operator.disposition,
        "admitted": operator.admitted,
        "kind": operator.kind,
    }


def _coerce_source_edit(
    raw: Mapping[str, Any] | AdmittedSourceEditOperator | None,
) -> AdmittedSourceEditOperator | None:
    if raw is None:
        return None
    if isinstance(raw, AdmittedSourceEditOperator):
        raw = _source_edit_mapping(raw)
    if not isinstance(raw, Mapping):
        raise RepairControllerError("source_edit_operator must be an object")
    _reject_forbidden_keys(raw, "source_edit_operator")
    try:
        return AdmittedSourceEditOperator.from_mapping(raw)
    except AdmittedSourceEditError as exc:
        raise RepairControllerError(f"source_edit_not_admitted:{exc}") from exc


def _bind_symbols(attempt: RepairAttempt) -> tuple[str, ...]:
    reasons: list[str] = []
    allowed = set(attempt.envelope.allowed_symbols)
    forbidden = set(attempt.plan.forbidden_symbols)
    for symbol in attempt.proposed_symbols:
        if forbidden and symbol in forbidden:
            reasons.append("forbidden_symbol")
        if allowed and symbol not in allowed:
            reasons.append("symbol_not_in_envelope")
    unique: list[str] = []
    for item in reasons:
        if item not in unique:
            unique.append(item)
    return tuple(unique)


def _model_assisted_gaps(attempt: RepairAttempt) -> tuple[str, ...]:
    plan = attempt.plan
    envelope = attempt.envelope
    gaps: list[str] = []
    if not plan.worktree_id:
        gaps.append("isolated_worktree_required")
    if not attempt.isolated_worktree_root and not plan.worktree_id:
        gaps.append("isolated_worktree_root_required")
    if not plan.predicted_files or not plan.predicted_symbols:
        gaps.append("exact_files_and_symbols_required")
    if not plan.context_reference_ids:
        gaps.append("sufficient_context_required")
    if envelope.required_test_ids and not plan.required_test_ids:
        gaps.append("predetermined_tests_required")
    if envelope.required_proof_ids and not plan.required_proof_ids:
        gaps.append("predetermined_proofs_required")
    if not set(envelope.required_test_ids).issubset(plan.required_test_ids):
        gaps.append("predetermined_tests_required")
    if not set(envelope.required_proof_ids).issubset(plan.required_proof_ids):
        gaps.append("predetermined_proofs_required")
    unique: list[str] = []
    for item in gaps:
        if item not in unique:
            unique.append(item)
    return tuple(unique)


def select_repair_tier(attempt: RepairAttempt) -> tuple[RepairTier, tuple[str, ...]]:
    """Select the cheapest sufficient tier; never silently upgrade to a model."""

    requested = attempt.plan.repair_tier
    if requested is RepairTier.MODEL_ASSISTED_BOUNDED:
        gaps = _model_assisted_gaps(attempt)
        if gaps:
            return requested, gaps
        return requested, ()
    if requested is RepairTier.TEMPLATE_CONSTRAINED:
        if not attempt.plan.context_reference_ids:
            return requested, ("template_context_required",)
        return requested, ()
    return RepairTier.DETERMINISTIC, ()


def admit_repair_attempt(attempt: RepairAttempt) -> RepairAdmission:
    """Admit or reject one attempt against envelope, scope, and authority paths."""

    selected_tier, tier_gaps = select_repair_tier(attempt)
    paths = attempt.proposed_paths
    violation, path_reasons = classify_paths(paths, allowed_paths=attempt.envelope.allowed_paths)
    reasons = list(path_reasons)
    reasons.extend(_bind_symbols(attempt))
    reasons.extend(tier_gaps)
    source_edit = None
    if attempt.source_edit_operator is not None:
        try:
            source_edit = _coerce_source_edit(attempt.source_edit_operator)
        except RepairControllerError as exc:
            reasons.append("source_edit_not_admitted")
            message = str(exc)
            if message not in reasons:
                reasons.append(message.split(":", 1)[0])
        else:
            edit_path = source_edit.relative_path
            edit_violation = classify_repair_path(
                edit_path, allowed_paths=attempt.envelope.allowed_paths
            )
            if edit_violation is not RepairPathViolation.NONE:
                violation = edit_violation if violation is RepairPathViolation.NONE else violation
                code = _reason_code(edit_violation)
                if code not in reasons:
                    reasons.append(code)
            if edit_path not in attempt.plan.predicted_files:
                reasons.append("source_edit_path_not_predicted")
            if not _path_in_prefixes(edit_path, attempt.plan.allowed_paths):
                reasons.append("source_edit_path_not_allowed")
            if attempt.isolated_worktree_root:
                try:
                    source_edit.validate(
                        repo_root=Path(attempt.isolated_worktree_root).resolve(),
                        preferred_path=edit_path,
                    )
                except AdmittedSourceEditError as exc:
                    code = str(exc).replace(" ", "_")
                    if not code.startswith("source_edit_"):
                        code = f"source_edit_{code}"
                    if code not in reasons:
                        reasons.append(code)
    if attempt.apply_source_edit and source_edit is None:
        if "source_edit_not_admitted" not in reasons:
            reasons.append("source_edit_not_admitted")
    if attempt.plan.risk_class.rank > attempt.envelope.risk_assessment.risk_class.rank:
        reasons.append("plan_raises_risk")
    if not attempt.policy.allows(
        attempt.envelope.autonomy_level, attempt.envelope.risk_assessment.risk_class
    ):
        reasons.append("autonomy_level_not_allowed")
    unique: list[str] = []
    for item in reasons:
        if item not in unique:
            unique.append(item)
    admitted = not unique and violation is RepairPathViolation.NONE
    return RepairAdmission(
        admitted=admitted,
        violation=violation,
        reason_codes=tuple(unique) if unique else ("admitted",),
        selected_tier=selected_tier,
        proposed_paths=paths,
        proposed_symbols=attempt.proposed_symbols,
        source_edit=source_edit,
    )


@dataclass
class _FailureRecord:
    signature: str
    count: int
    diagnostic_receipt_id: str
    last_backoff_ms: int


class AutonomousRepairController:
    """Tier-selection and scope facade over ``AutonomousRepairEngine``.

    The controller never subclasses or replaces the engine.  Model-assisted
    repair may invoke an injected callable at most once per distinct failure
    signature; identical failures reuse the prior diagnosis and back off.
    """

    INTERFACE: Final[str] = AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    def __init__(
        self,
        *,
        engine: AutonomousRepairEngine | None = None,
        repo_root: str | Path | None = None,
        model_invoker: Callable[[RepairAttempt], Mapping[str, Any]] | None = None,
        decision_runtime: Any | None = None,
    ) -> None:
        if engine is None:
            if repo_root is None:
                raise RepairControllerError("AutonomousRepairEngine or repo_root is required")
            engine = AutonomousRepairEngine(repo_root=repo_root)
        if not isinstance(engine, AutonomousRepairEngine):
            raise RepairControllerError("engine must be AutonomousRepairEngine")
        if type(self) is not AutonomousRepairController and issubclass(
            type(self), AutonomousRepairEngine
        ):
            raise RepairControllerError("controller cannot be a second repair engine")
        self._engine = engine
        self._model_invoker = model_invoker
        self._decision_runtime = decision_runtime
        self._failures: dict[str, _FailureRecord] = {}

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine(self) -> AutonomousRepairEngine:
        return self._engine

    @property
    def decision_runtime(self) -> Any | None:
        return self._decision_runtime

    def failure_count(self, signature: str) -> int:
        record = self._failures.get(signature)
        return 0 if record is None else record.count

    def select_tier(self, attempt: RepairAttempt) -> RepairTier:
        tier, _gaps = select_repair_tier(attempt)
        return tier

    def admit(self, attempt: RepairAttempt) -> RepairAdmission:
        return admit_repair_attempt(attempt)

    def admit_source_edit(
        self,
        operator: Mapping[str, Any] | AdmittedSourceEditOperator,
        *,
        envelope: AutonomyEnvelope,
        plan: AutonomousRepairPlan,
        policy: AutonomyPolicy,
        isolated_worktree_root: str = "",
    ) -> RepairAdmission:
        attempt = RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=plan,
            source_edit_operator=operator,
            isolated_worktree_root=isolated_worktree_root,
        )
        return self.admit(attempt)

    def merge_eligibility(
        self,
        attempt: RepairAttempt,
        *,
        admission: RepairAdmission | None = None,
        terminal_status: TerminalStatus = TerminalStatus.SUCCEEDED,
        changed_paths: Sequence[str] | None = None,
        validation_receipt_ids: Sequence[str] | None = None,
        proof_receipt_ids: Sequence[str] | None = None,
        backoff: bool = False,
    ) -> RepairMergeEligibility:
        admission = admission or self.admit(attempt)
        return evaluate_merge_eligibility(
            attempt,
            admission=admission,
            terminal_status=terminal_status,
            changed_paths=attempt.changed_paths if changed_paths is None else changed_paths,
            validation_receipt_ids=(
                attempt.validation_receipt_ids
                if validation_receipt_ids is None
                else validation_receipt_ids
            ),
            proof_receipt_ids=(
                attempt.proof_receipt_ids if proof_receipt_ids is None else proof_receipt_ids
            ),
            backoff=backoff,
        )

    def _failure_key(self, attempt: RepairAttempt) -> str:
        if attempt.failure_signature:
            return attempt.failure_signature
        return content_identity(
            {
                "schema": REPAIR_FAILURE_RECORD_SCHEMA,
                "envelope_id": attempt.envelope.envelope_id,
                "plan_id": attempt.plan.plan_id,
                "predicted_files": list(attempt.plan.predicted_files),
                "predicted_symbols": list(attempt.plan.predicted_symbols),
                "diagnostic_receipt_id": attempt.diagnostic_receipt_id,
            }
        )

    def _record_failure(self, signature: str, diagnostic_receipt_id: str) -> _FailureRecord:
        existing = self._failures.get(signature)
        count = 1 if existing is None else existing.count + 1
        if len(self._failures) >= MAX_FAILURE_RECORDS and existing is None:
            oldest = next(iter(self._failures))
            del self._failures[oldest]
        record = _FailureRecord(
            signature=signature,
            count=count,
            diagnostic_receipt_id=diagnostic_receipt_id or (existing.diagnostic_receipt_id if existing else ""),
            last_backoff_ms=_backoff_ms(count),
        )
        self._failures[signature] = record
        return record

    def _identical_failure(self, attempt: RepairAttempt) -> _FailureRecord | None:
        key = self._failure_key(attempt)
        record = self._failures.get(key)
        if record is None:
            return None
        if attempt.new_evidence_since_failure:
            return None
        return record

    def _receipt(
        self,
        attempt: RepairAttempt,
        *,
        status: TerminalStatus,
        changed_paths: Sequence[str],
        validation_receipt_ids: Sequence[str],
        proof_receipt_ids: Sequence[str],
        failure_signature: str = "",
        diagnostic_receipt_id: str = "",
        rollback_receipt_id: str = "",
    ) -> AutonomousRepairReceipt:
        return AutonomousRepairReceipt(
            plan_id=attempt.plan.plan_id,
            envelope_id=attempt.envelope.envelope_id,
            terminal_status=status,
            changed_paths=tuple(changed_paths),
            validation_receipt_ids=tuple(validation_receipt_ids),
            proof_receipt_ids=tuple(proof_receipt_ids),
            adversarial_assurance_receipt_ids=attempt.adversarial_assurance_receipt_ids,
            rollback_receipt_id=rollback_receipt_id,
            failure_signature=failure_signature,
            diagnostic_receipt_id=diagnostic_receipt_id,
            authorizes_merge=False,
        )

    def run(self, attempt: RepairAttempt) -> RepairControllerResult:
        admission = self.admit(attempt)
        selected_tier = admission.selected_tier
        if not admission.admitted:
            status = TerminalStatus.BLOCKED
            receipt = self._receipt(
                attempt,
                status=status,
                changed_paths=(),
                validation_receipt_ids=(),
                proof_receipt_ids=(),
                failure_signature=attempt.failure_signature,
                diagnostic_receipt_id=attempt.diagnostic_receipt_id,
            )
            merge = evaluate_merge_eligibility(
                attempt,
                admission=admission,
                terminal_status=status,
                changed_paths=(),
                validation_receipt_ids=(),
                proof_receipt_ids=(),
            )
            return RepairControllerResult(
                disposition=RepairControllerDisposition.REJECTED,
                admission=admission,
                merge=merge,
                receipt=receipt,
                selected_tier=selected_tier,
                reason_codes=admission.reason_codes,
                model_call_count=0,
                engine_invoked=False,
            )

        identical = self._identical_failure(attempt)
        if identical is not None and selected_tier is RepairTier.MODEL_ASSISTED_BOUNDED:
            diagnostic = identical.diagnostic_receipt_id or attempt.diagnostic_receipt_id
            receipt = self._receipt(
                attempt,
                status=TerminalStatus.EXHAUSTED,
                changed_paths=(),
                validation_receipt_ids=attempt.validation_receipt_ids,
                proof_receipt_ids=attempt.proof_receipt_ids,
                failure_signature=identical.signature,
                diagnostic_receipt_id=diagnostic,
            )
            merge = evaluate_merge_eligibility(
                attempt,
                admission=admission,
                terminal_status=TerminalStatus.EXHAUSTED,
                changed_paths=(),
                validation_receipt_ids=attempt.validation_receipt_ids,
                proof_receipt_ids=attempt.proof_receipt_ids,
                backoff=True,
            )
            return RepairControllerResult(
                disposition=RepairControllerDisposition.BACKOFF,
                admission=admission,
                merge=merge,
                receipt=receipt,
                selected_tier=selected_tier,
                reason_codes=("identical_failure_backoff", "diagnosis_reused"),
                model_call_count=0,
                diagnostic_reused=True,
                backoff_milliseconds=identical.last_backoff_ms,
                engine_invoked=False,
            )

        engine_report = self._engine.run(attempt.work_items)
        engine_payload = _engine_report_payload(engine_report)
        model_calls = int(getattr(engine_report, "model_call_count", 0) or 0)
        if selected_tier is RepairTier.MODEL_ASSISTED_BOUNDED and self._model_invoker is not None:
            invoked = self._model_invoker(attempt)
            if not isinstance(invoked, Mapping):
                raise RepairControllerError("model invoker must return a mapping")
            model_calls += int(invoked.get("model_call_count") or 1)

        changed_paths = attempt.changed_paths or attempt.plan.predicted_files
        validation_ids = attempt.validation_receipt_ids
        proof_ids = attempt.proof_receipt_ids
        succeeded = bool(validation_ids) and not getattr(engine_report, "llm_used", False)
        if selected_tier is RepairTier.MODEL_ASSISTED_BOUNDED and model_calls and not validation_ids:
            succeeded = False
        status = TerminalStatus.SUCCEEDED if succeeded else TerminalStatus.FAILED
        failure_signature = ""
        diagnostic = attempt.diagnostic_receipt_id
        if status is TerminalStatus.FAILED:
            failure_signature = self._failure_key(attempt)
            record = self._record_failure(failure_signature, diagnostic)
            diagnostic = record.diagnostic_receipt_id or diagnostic
        receipt = self._receipt(
            attempt,
            status=status,
            changed_paths=changed_paths if succeeded else (),
            validation_receipt_ids=validation_ids if succeeded else validation_ids,
            proof_receipt_ids=proof_ids,
            failure_signature=failure_signature,
            diagnostic_receipt_id=diagnostic,
        )
        merge = evaluate_merge_eligibility(
            attempt,
            admission=admission,
            terminal_status=status,
            changed_paths=changed_paths if succeeded else (),
            validation_receipt_ids=validation_ids,
            proof_receipt_ids=proof_ids,
        )
        if merge.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE:
            disposition = RepairControllerDisposition.AUTONOMOUS_MERGE
        elif merge.disposition is RepairMergeDisposition.PROPOSAL:
            disposition = RepairControllerDisposition.PROPOSAL
        elif status is TerminalStatus.SUCCEEDED:
            disposition = RepairControllerDisposition.EXECUTED
        else:
            disposition = RepairControllerDisposition.REJECTED
        reasons = ["engine_composed"]
        if attempt.apply_source_edit:
            reasons.append("source_edit_not_applied_by_controller")
        if merge.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE:
            reasons.append("low_risk_merge_conjunction")
        elif merge.disposition is RepairMergeDisposition.PROPOSAL:
            reasons.append("r3_proposal")
        if status is TerminalStatus.FAILED:
            reasons.append("repair_failed")
        return RepairControllerResult(
            disposition=disposition,
            admission=admission,
            merge=merge,
            receipt=receipt,
            selected_tier=selected_tier,
            reason_codes=tuple(reasons),
            model_call_count=model_calls,
            engine_invoked=True,
            engine_report=engine_payload,
        )


__all__ = [
    "AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE",
    "AUTONOMOUS_REPAIR_CONTROLLER_SCHEMA",
    "CONTROLLER_RELATIVE_PATH",
    "LOW_RISK_MERGE_CONDITIONS",
    "PROTECTED_AUTHORITY_PATHS",
    "SELF_EDIT_PATHS",
    "AdmittedSourceEditOperator",
    "AutonomousRepairController",
    "AutonomousRepairEngine",
    "RepairAdmission",
    "RepairAttempt",
    "RepairControllerDisposition",
    "RepairControllerError",
    "RepairControllerResult",
    "RepairMergeDisposition",
    "RepairMergeEligibility",
    "RepairPathViolation",
    "admit_repair_attempt",
    "classify_repair_path",
    "evaluate_merge_eligibility",
    "is_protected_authority_path",
    "is_self_edit_path",
    "is_validator_policy_key_path",
    "select_repair_tier",
]
