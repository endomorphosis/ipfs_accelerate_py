# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded facade over the existing autonomous-repair engine.

``AutonomousRepairController@1`` selects a repair tier, binds one exact
envelope, and refuses scope escape, self-edit, and validator/policy/key
mutation.  It is not a second repair engine, materializer, merge authority,
or ``DecisionRuntime``.  Admission and execution stay with
``AutonomousRepairEngine`` and the existing source-edit operator; this
module only composes those authorities and records merge *eligibility*.

Model-assisted repair is last.  It requires exact files/symbols, a bounded
patch envelope, sufficient context, an isolated worktree, and predetermined
tests/proofs.  Identical failures reuse the prior diagnostic and never
repeat a model call.  Autonomous merge is a conjunction over every stated
low-risk condition; R3 remains a proposal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
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
    AutonomyContractError,
    AutonomyEnvelope,
    AutonomyLevel,
    AutonomyPolicy,
    RepairTier,
    RiskClass,
    TerminalStatus,
)

AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE: Final[str] = "AutonomousRepairController@1"
AUTONOMOUS_REPAIR_CONTROLLER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/autonomous-repair-controller@1"
)
REPAIR_CONTROLLER_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-attempt@1"
)
REPAIR_CONTROLLER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-result@1"
)
REPAIR_FAILURE_MEMORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-failure-memory@1"
)

MAX_REPAIR_CONTROLLER_SNAPSHOT_BYTES: Final[int] = 4 * MAX_CANONICAL_RECORD_BYTES
DEFAULT_IDENTICAL_FAILURE_LIMIT: Final[int] = 3
DEFAULT_BASE_BACKOFF_MS: Final[int] = 1_000
DEFAULT_MAX_BACKOFF_MS: Final[int] = 60_000
DEFAULT_MAX_CHANGED_FILES: Final[int] = 8
DEFAULT_MAX_CHANGED_LINES: Final[int] = 400

SELF_EDIT_PATHS: Final[frozenset[str]] = frozenset(
    {
        "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",
    }
)

# Operator-protected and other self-protecting authority surfaces.  Repair may
# never mutate these even when an envelope lists a parent directory.
PROTECTED_AUTHORITY_PATHS: Final[frozenset[str]] = frozenset(
    {
        ".gitignore",
        "config/agent_supervisor_autonomous_meta_controller_scheduler.json",
        "docs/architecture/AGENT_SUPERVISOR_AUTONOMOUS_META_CONTROLLER_PLAN.md",
        "docs/architecture/agent_supervisor_autonomous_meta_controller.objectives.md",
        "docs/architecture/agent_supervisor_autonomous_meta_controller.todo.md",
        "ipfs_accelerate_py/agent_implementation_route.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_catalog.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/mcp_invocation_trace.py",
        "ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",
        "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",
        "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
        "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py",
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
        "scripts/lgswf_start_quack_control.py",
        "scripts/materialize_agent_supervisor_autonomous_meta_controller_board.py",
        "scripts/ops/agent_supervisor/quack_state_server.py",
        "scripts/validate_agent_supervisor_autonomous_meta_controller_board.py",
    }
)

VALIDATOR_POLICY_KEY_PATHS: Final[frozenset[str]] = frozenset(
    {
        "config/agent_supervisor_autonomous_meta_controller_scheduler.json",
        "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
        "ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "ipfs_accelerate_py/llm_router.py",
    }
)

VALIDATOR_POLICY_KEY_MARKERS: Final[tuple[str, ...]] = (
    "policy_key",
    "promotion_rule",
    "trusted_key",
    "validator_policy",
)

DEFAULT_FORBIDDEN_SYMBOLS: Final[tuple[str, ...]] = (
    "policy_key",
    "promotion_rule",
    "trusted_keys",
    "validator_policy",
)

# Named conjunction used by autonomous merge.  Every member must hold for R2
# (or lower) isolated repair.  R3 is proposal-only even when others hold.
LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "risk_at_most_r2",
    "reversible",
    "autonomous_merge_enabled",
    "autonomy_level_permits_execution",
    "policy_allows_requested_level",
    "repair_succeeded",
    "required_tests_satisfied",
    "required_proofs_satisfied",
    "isolated_worktree_bound",
    "rollback_plan_bound",
    "changed_paths_within_envelope",
    "no_protected_authority_path",
    "no_self_edit",
    "no_validator_policy_key_mutation",
    "patch_bounds_respected",
    "predetermined_checks_complete",
)

_EXECUTING_LEVELS: Final[frozenset[AutonomyLevel]] = frozenset(
    {
        AutonomyLevel.EXECUTE_REVERSIBLE,
        AutonomyLevel.EXECUTE_BOUNDED_MUTATION,
        AutonomyLevel.SELF_REPAIR_ISOLATED,
    }
)
_MERGE_ELIGIBLE_RISK: Final[frozenset[RiskClass]] = frozenset(
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
    """Closed outcome of one facade step.  None of these is merge authority."""

    ADMITTED = "admitted"
    EXECUTED = "executed"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    IDENTICAL_FAILURE_BACKOFF = "identical_failure_backoff"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    ROLLBACK = "rollback"
    PROPOSAL = "proposal"
    AUTONOMOUS_MERGE_ELIGIBLE = "autonomous_merge_eligible"


class RepairMergeDisposition(str, Enum):
    """Merge *eligibility* only.  Existing merge authorities remain canonical."""

    NONE = "none"
    REJECTED = "rejected"
    ROLLBACK = "rollback"
    PROPOSAL = "proposal"
    AUTONOMOUS_MERGE = "autonomous_merge"


ModelAssistedRepairHook = Callable[
    ["AutonomousRepairPlan", "AutonomousRepairAttempt"],
    Mapping[str, Any] | None,
]


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


def _posix_path(value: Any, name: str) -> str:
    result = _identifier(value, name)
    parsed = PurePosixPath(result)
    if "\\" in result or parsed.is_absolute() or ".." in parsed.parts or result in {".", ""}:
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
        raise RepairControllerError(
            f"{name} must be an integer between {minimum} and {maximum}"
        )
    return value


def _reject_forbidden_keys(payload: Mapping[str, Any], name: str) -> None:
    if len(payload) > MAX_MAPPING_ITEMS:
        raise RepairControllerError(f"{name} contains too many fields")
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


def path_is_within(path: str, prefixes: Sequence[str]) -> bool:
    """Return True when ``path`` equals or is a descendant of any prefix."""

    candidate = PurePosixPath(path).as_posix()
    for prefix in prefixes:
        bound = PurePosixPath(prefix).as_posix().rstrip("/")
        if candidate == bound or candidate.startswith(bound + "/"):
            return True
    return False


def is_protected_authority_path(path: str) -> bool:
    return path_is_within(path, tuple(sorted(PROTECTED_AUTHORITY_PATHS)))


def is_self_edit_path(path: str) -> bool:
    return path_is_within(path, tuple(sorted(SELF_EDIT_PATHS)))


def is_validator_policy_key_path(path: str) -> bool:
    if path_is_within(path, tuple(sorted(VALIDATOR_POLICY_KEY_PATHS))):
        return True
    lowered = path.lower().replace("-", "_")
    return any(marker in lowered for marker in VALIDATOR_POLICY_KEY_MARKERS)


def _as_reason(value: str) -> str:
    compact = "_".join(str(value).strip().lower().replace(":", " ").replace("/", " ").split())
    if not compact:
        return "rejected"
    encoded = compact.encode("utf-8")
    if len(encoded) > MAX_IDENTIFIER_BYTES:
        compact = encoded[:MAX_IDENTIFIER_BYTES].decode("utf-8", errors="ignore").rstrip("_")
    return compact or "rejected"


def _classify_forbidden_paths(paths: Sequence[str]) -> tuple[str, ...]:
    reasons: list[str] = []
    if any(is_self_edit_path(path) for path in paths):
        reasons.append("self_edit_rejected")
    if any(is_validator_policy_key_path(path) for path in paths):
        reasons.append("validator_policy_key_mutation_rejected")
    if any(is_protected_authority_path(path) for path in paths):
        reasons.append("protected_authority_path_rejected")
    return tuple(reasons)


def _coerce_envelope(value: Any) -> AutonomyEnvelope:
    if isinstance(value, AutonomyEnvelope):
        return value
    if isinstance(value, Mapping):
        try:
            return AutonomyEnvelope.from_dict(value)
        except AutonomyContractError as exc:
            raise RepairControllerError(str(exc)) from exc
    raise RepairControllerError("envelope must be an AutonomyEnvelope")


def _coerce_policy(value: Any) -> AutonomyPolicy:
    if isinstance(value, AutonomyPolicy):
        return value
    if isinstance(value, Mapping):
        try:
            return AutonomyPolicy.from_dict(value)
        except AutonomyContractError as exc:
            raise RepairControllerError(str(exc)) from exc
    raise RepairControllerError("policy must be an AutonomyPolicy")


@dataclass(frozen=True)
class RepairFailureRecord:
    """One identical-failure signature and its reusable diagnostic."""

    failure_signature: str
    diagnostic_receipt_id: str
    count: int
    backoff_milliseconds: int
    model_call_count: int
    exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_signature": self.failure_signature,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "count": self.count,
            "backoff_milliseconds": self.backoff_milliseconds,
            "model_call_count": self.model_call_count,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True)
class AutonomousRepairAttempt:
    """One bounded repair request.  Never a prompt, transcript, or patch body."""

    envelope: AutonomyEnvelope
    policy: AutonomyPolicy
    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    worktree_id: str = ""
    rollback_plan_id: str = ""
    context_reference_ids: tuple[str, ...] = ()
    required_test_ids: tuple[str, ...] = ()
    required_proof_ids: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    forbidden_symbols: tuple[str, ...] = DEFAULT_FORBIDDEN_SYMBOLS
    max_changed_files: int = DEFAULT_MAX_CHANGED_FILES
    max_changed_lines: int = DEFAULT_MAX_CHANGED_LINES
    requested_tier: RepairTier | None = None
    template_id: str = ""
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""
    changed_paths: tuple[str, ...] = ()
    changed_line_count: int = 0
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    source_edit_operator: Mapping[str, Any] | None = None
    now_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelope", _coerce_envelope(self.envelope))
        object.__setattr__(self, "policy", _coerce_policy(self.policy))
        object.__setattr__(
            self, "predicted_files", _posix_paths(self.predicted_files, "predicted_files")
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _identifiers(self.predicted_symbols, "predicted_symbols", required=True),
        )
        object.__setattr__(
            self, "worktree_id", _identifier(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self,
            "rollback_plan_id",
            _identifier(self.rollback_plan_id, "rollback_plan_id", required=False),
        )
        for name in (
            "context_reference_ids",
            "required_test_ids",
            "required_proof_ids",
            "validation_receipt_ids",
            "proof_receipt_ids",
            "adversarial_assurance_receipt_ids",
        ):
            object.__setattr__(
                self,
                name,
                _identifiers(getattr(self, name), name),
            )
        allowed = self.allowed_paths or self.envelope.allowed_paths
        object.__setattr__(
            self, "allowed_paths", _posix_paths(allowed, "allowed_paths", required=True)
        )
        forbidden = self.forbidden_symbols or DEFAULT_FORBIDDEN_SYMBOLS
        object.__setattr__(
            self, "forbidden_symbols", _identifiers(forbidden, "forbidden_symbols")
        )
        object.__setattr__(
            self,
            "max_changed_files",
            _int(self.max_changed_files, "max_changed_files", minimum=1, maximum=10_000),
        )
        object.__setattr__(
            self,
            "max_changed_lines",
            _int(self.max_changed_lines, "max_changed_lines", minimum=1, maximum=1_000_000),
        )
        if self.requested_tier is not None:
            object.__setattr__(
                self,
                "requested_tier",
                _enum(self.requested_tier, RepairTier, "requested_tier"),
            )
        object.__setattr__(
            self, "template_id", _identifier(self.template_id, "template_id", required=False)
        )
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
            self, "changed_paths", _posix_paths(self.changed_paths, "changed_paths")
        )
        object.__setattr__(
            self,
            "changed_line_count",
            _int(self.changed_line_count, "changed_line_count", maximum=1_000_000),
        )
        if self.source_edit_operator is not None:
            if not isinstance(self.source_edit_operator, Mapping):
                raise RepairControllerError("source_edit_operator must be an object")
            _reject_forbidden_keys(self.source_edit_operator, "source_edit_operator")
            object.__setattr__(
                self,
                "source_edit_operator",
                MappingProxyType(dict(self.source_edit_operator)),
            )
        object.__setattr__(self, "now_ms", _int(self.now_ms, "now_ms"))
        tests = self.required_test_ids or self.envelope.required_test_ids
        proofs = self.required_proof_ids or self.envelope.required_proof_ids
        object.__setattr__(self, "required_test_ids", tests)
        object.__setattr__(self, "required_proof_ids", proofs)
        if not self.predicted_files:
            raise RepairControllerError("predicted_files must not be empty")
        if len(self.predicted_files) > self.max_changed_files:
            raise RepairControllerError("predicted repair files exceed the patch envelope")

    @property
    def mutation_paths(self) -> tuple[str, ...]:
        return self.changed_paths or self.predicted_files

    @property
    def patch_envelope_id(self) -> str:
        return self.envelope.envelope_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_CONTROLLER_ATTEMPT_SCHEMA,
            "envelope_id": self.envelope.envelope_id,
            "policy_id": self.policy.policy_id,
            "predicted_files": list(self.predicted_files),
            "predicted_symbols": list(self.predicted_symbols),
            "worktree_id": self.worktree_id,
            "rollback_plan_id": self.rollback_plan_id,
            "context_reference_ids": list(self.context_reference_ids),
            "required_test_ids": list(self.required_test_ids),
            "required_proof_ids": list(self.required_proof_ids),
            "allowed_paths": list(self.allowed_paths),
            "forbidden_symbols": list(self.forbidden_symbols),
            "max_changed_files": self.max_changed_files,
            "max_changed_lines": self.max_changed_lines,
            "requested_tier": None if self.requested_tier is None else self.requested_tier.value,
            "template_id": self.template_id,
            "failure_signature": self.failure_signature,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "changed_paths": list(self.changed_paths),
            "changed_line_count": self.changed_line_count,
            "validation_receipt_ids": list(self.validation_receipt_ids),
            "proof_receipt_ids": list(self.proof_receipt_ids),
            "adversarial_assurance_receipt_ids": list(self.adversarial_assurance_receipt_ids),
            "source_edit_operator": (
                None if self.source_edit_operator is None else dict(self.source_edit_operator)
            ),
            "now_ms": self.now_ms,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | AutonomousRepairAttempt) -> AutonomousRepairAttempt:
        if isinstance(payload, AutonomousRepairAttempt):
            return payload
        if not isinstance(payload, Mapping):
            raise RepairControllerError("repair attempt must be an object")
        _reject_forbidden_keys(payload, "repair attempt")
        return cls(
            envelope=payload.get("envelope"),
            policy=payload.get("policy"),
            predicted_files=tuple(payload.get("predicted_files") or ()),
            predicted_symbols=tuple(payload.get("predicted_symbols") or ()),
            worktree_id=str(payload.get("worktree_id") or ""),
            rollback_plan_id=str(payload.get("rollback_plan_id") or ""),
            context_reference_ids=tuple(payload.get("context_reference_ids") or ()),
            required_test_ids=tuple(payload.get("required_test_ids") or ()),
            required_proof_ids=tuple(payload.get("required_proof_ids") or ()),
            allowed_paths=tuple(payload.get("allowed_paths") or ()),
            forbidden_symbols=tuple(
                payload.get("forbidden_symbols")
                if payload.get("forbidden_symbols") is not None
                else DEFAULT_FORBIDDEN_SYMBOLS
            ),
            max_changed_files=int(
                payload.get("max_changed_files") or DEFAULT_MAX_CHANGED_FILES
            ),
            max_changed_lines=int(
                payload.get("max_changed_lines") or DEFAULT_MAX_CHANGED_LINES
            ),
            requested_tier=payload.get("requested_tier"),
            template_id=str(payload.get("template_id") or ""),
            failure_signature=str(payload.get("failure_signature") or ""),
            diagnostic_receipt_id=str(payload.get("diagnostic_receipt_id") or ""),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            changed_line_count=int(payload.get("changed_line_count") or 0),
            validation_receipt_ids=tuple(payload.get("validation_receipt_ids") or ()),
            proof_receipt_ids=tuple(payload.get("proof_receipt_ids") or ()),
            adversarial_assurance_receipt_ids=tuple(
                payload.get("adversarial_assurance_receipt_ids") or ()
            ),
            source_edit_operator=payload.get("source_edit_operator"),
            now_ms=int(payload.get("now_ms") or 0),
        )


def _scope_reasons(attempt: AutonomousRepairAttempt) -> tuple[str, ...]:
    reasons: list[str] = []
    envelope_paths = attempt.envelope.allowed_paths
    allowed = attempt.allowed_paths
    mutation = attempt.mutation_paths
    if not all(path_is_within(path, envelope_paths) for path in mutation):
        reasons.append("scope_escape_rejected")
    if not all(path_is_within(path, allowed) for path in mutation):
        reasons.append("scope_escape_rejected")
    if not all(path_is_within(path, allowed) for path in attempt.predicted_files):
        reasons.append("scope_escape_rejected")
    reasons.extend(_classify_forbidden_paths((*attempt.predicted_files, *mutation)))
    forbidden = set(attempt.forbidden_symbols)
    if any(symbol in forbidden for symbol in attempt.predicted_symbols):
        reasons.append("forbidden_symbol_rejected")
    allowed_symbols = attempt.envelope.allowed_symbols
    if allowed_symbols and not set(attempt.predicted_symbols).issubset(set(allowed_symbols)):
        reasons.append("symbol_escape_rejected")
    if attempt.now_ms and attempt.envelope.expiry_ms and attempt.now_ms > attempt.envelope.expiry_ms:
        reasons.append("envelope_expired")
    # Preserve first-seen order while dropping duplicates.
    unique: list[str] = []
    seen: set[str] = set()
    for item in reasons:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)


def select_repair_tier(attempt: AutonomousRepairAttempt) -> RepairTier:
    """Choose the cheapest admissible tier.  Model-assisted is last."""

    if attempt.requested_tier is not None:
        return attempt.requested_tier
    if attempt.template_id:
        return RepairTier.TEMPLATE_CONSTRAINED
    return RepairTier.DETERMINISTIC


def _plan_construction_gaps(attempt: AutonomousRepairAttempt, tier: RepairTier) -> tuple[str, ...]:
    gaps: list[str] = []
    if not attempt.predicted_files:
        gaps.append("exact_files_required")
    if not attempt.predicted_symbols:
        gaps.append("exact_symbols_required")
    if not attempt.worktree_id:
        gaps.append("isolated_worktree_required")
    if not attempt.context_reference_ids:
        gaps.append("sufficient_context_required")
    if not attempt.rollback_plan_id:
        gaps.append("rollback_plan_required")
    if tier is RepairTier.MODEL_ASSISTED_BOUNDED and (
        not attempt.required_test_ids and not attempt.required_proof_ids
    ):
        gaps.append("predetermined_checks_required")
    return tuple(gaps)


def evaluate_low_risk_merge_conditions(
    attempt: AutonomousRepairAttempt,
    *,
    terminal_status: TerminalStatus,
    plan: AutonomousRepairPlan | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(satisfied, missing)`` for the closed R2 merge conjunction."""

    risk = attempt.envelope.risk_assessment.risk_class
    level = attempt.envelope.autonomy_level
    mutation = attempt.mutation_paths
    checks: dict[str, bool] = {
        "risk_at_most_r2": risk in _MERGE_ELIGIBLE_RISK,
        "reversible": bool(attempt.envelope.reversible and attempt.envelope.risk_assessment.reversible),
        "autonomous_merge_enabled": bool(attempt.policy.autonomous_merge_enabled),
        "autonomy_level_permits_execution": level in _EXECUTING_LEVELS,
        "policy_allows_requested_level": attempt.policy.allows(level, risk),
        "repair_succeeded": terminal_status is TerminalStatus.SUCCEEDED,
        "required_tests_satisfied": (
            not attempt.required_test_ids or bool(attempt.validation_receipt_ids)
        ),
        "required_proofs_satisfied": (
            not attempt.required_proof_ids or bool(attempt.proof_receipt_ids)
        ),
        "isolated_worktree_bound": bool(attempt.worktree_id),
        "rollback_plan_bound": bool(attempt.rollback_plan_id),
        "changed_paths_within_envelope": all(
            path_is_within(path, attempt.allowed_paths)
            and path_is_within(path, attempt.envelope.allowed_paths)
            for path in mutation
        ),
        "no_protected_authority_path": not any(
            is_protected_authority_path(path) for path in mutation
        ),
        "no_self_edit": not any(is_self_edit_path(path) for path in mutation),
        "no_validator_policy_key_mutation": not any(
            is_validator_policy_key_path(path) for path in mutation
        ),
        "patch_bounds_respected": (
            len(mutation) <= attempt.max_changed_files
            and attempt.changed_line_count <= attempt.max_changed_lines
            and (plan is None or len(plan.predicted_files) <= plan.max_changed_files)
        ),
        "predetermined_checks_complete": bool(attempt.validation_receipt_ids)
        and (not attempt.required_proof_ids or bool(attempt.proof_receipt_ids)),
    }
    satisfied = tuple(name for name in LOW_RISK_MERGE_CONDITIONS if checks[name])
    missing = tuple(name for name in LOW_RISK_MERGE_CONDITIONS if not checks[name])
    return satisfied, missing


def _merge_disposition_for(
    attempt: AutonomousRepairAttempt,
    *,
    terminal_status: TerminalStatus,
    rejection_reasons: Sequence[str],
    satisfied: Sequence[str],
    missing: Sequence[str],
) -> RepairMergeDisposition:
    risk = attempt.envelope.risk_assessment.risk_class
    if rejection_reasons:
        return RepairMergeDisposition.REJECTED
    if terminal_status in {TerminalStatus.FAILED, TerminalStatus.BLOCKED}:
        if attempt.rollback_plan_id:
            return RepairMergeDisposition.ROLLBACK
        return RepairMergeDisposition.REJECTED
    if terminal_status in {
        TerminalStatus.EXHAUSTED,
        TerminalStatus.CANCELLED,
        TerminalStatus.UNAVAILABLE,
    }:
        return RepairMergeDisposition.REJECTED
    if risk is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL:
        return RepairMergeDisposition.REJECTED
    if risk is RiskClass.R3_BOUNDED_REPOSITORY_MUTATION:
        return RepairMergeDisposition.PROPOSAL
    if risk is RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE:
        return RepairMergeDisposition.PROPOSAL
    if (
        terminal_status is TerminalStatus.SUCCEEDED
        and not missing
        and tuple(satisfied) == LOW_RISK_MERGE_CONDITIONS
    ):
        return RepairMergeDisposition.AUTONOMOUS_MERGE
    if terminal_status is TerminalStatus.SUCCEEDED:
        return RepairMergeDisposition.PROPOSAL
    return RepairMergeDisposition.NONE


def admit_source_edit_operator(
    operator: Mapping[str, Any] | AdmittedSourceEditOperator | None,
    *,
    predicted_files: Sequence[str],
    allowed_paths: Sequence[str],
    envelope_paths: Sequence[str] | None = None,
) -> AdmittedSourceEditOperator:
    """Admit an exact source-edit operator without writing bytes."""

    if operator is None:
        raise AdmittedSourceEditError("source_edit_operator_missing")
    if isinstance(operator, AdmittedSourceEditOperator):
        admitted = operator
    else:
        admitted = AdmittedSourceEditOperator.from_mapping(operator)
    try:
        relative = _posix_path(admitted.relative_path, "source_edit_relative_path")
    except RepairControllerError as exc:
        raise RepairControllerError("scope_escape_rejected") from exc
    if relative not in set(predicted_files):
        raise RepairControllerError("source_edit_path_not_in_predicted_files")
    if not path_is_within(relative, allowed_paths):
        raise RepairControllerError("scope_escape_rejected")
    if envelope_paths is not None and not path_is_within(relative, envelope_paths):
        raise RepairControllerError("scope_escape_rejected")
    forbidden = _classify_forbidden_paths((relative,))
    if forbidden:
        raise RepairControllerError(forbidden[0])
    return admitted


@dataclass(frozen=True)
class AutonomousRepairControllerResult:
    """Facade receipt.  Evidence only; never merge or effect authority."""

    disposition: RepairControllerDisposition
    merge_disposition: RepairMergeDisposition
    selected_tier: RepairTier | None = None
    plan: AutonomousRepairPlan | None = None
    receipt: AutonomousRepairReceipt | None = None
    reason_codes: tuple[str, ...] = ()
    model_call_count: int = 0
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    satisfied_merge_conditions: tuple[str, ...] = ()
    missing_merge_conditions: tuple[str, ...] = ()
    source_edit_admitted: bool = False
    engine_report_id: str = ""
    diagnostic_receipt_id: str = ""
    failure_signature: str = ""

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
        if self.selected_tier is not None:
            object.__setattr__(
                self, "selected_tier", _enum(self.selected_tier, RepairTier, "selected_tier")
            )
        if self.plan is not None and not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan")
        if self.receipt is not None and not isinstance(self.receipt, AutonomousRepairReceipt):
            raise RepairControllerError("receipt must be an AutonomousRepairReceipt")
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", preserve_order=True),
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
        object.__setattr__(
            self,
            "satisfied_merge_conditions",
            _identifiers(
                self.satisfied_merge_conditions,
                "satisfied_merge_conditions",
                preserve_order=True,
            ),
        )
        object.__setattr__(
            self,
            "missing_merge_conditions",
            _identifiers(
                self.missing_merge_conditions,
                "missing_merge_conditions",
                preserve_order=True,
            ),
        )
        object.__setattr__(
            self,
            "source_edit_admitted",
            _bool(self.source_edit_admitted, "source_edit_admitted"),
        )
        object.__setattr__(
            self,
            "engine_report_id",
            _identifier(self.engine_report_id, "engine_report_id", required=False),
        )
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _identifier(self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False),
        )
        object.__setattr__(
            self,
            "failure_signature",
            _identifier(self.failure_signature, "failure_signature", required=False),
        )
        if self.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE:
            if tuple(self.satisfied_merge_conditions) != LOW_RISK_MERGE_CONDITIONS:
                raise RepairControllerError(
                    "autonomous merge requires every stated low-risk condition"
                )
            if self.missing_merge_conditions:
                raise RepairControllerError(
                    "autonomous merge cannot have missing low-risk conditions"
                )
        encoded = canonical_json(self.to_dict(include_identity=False)).encode("utf-8")
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise RepairControllerError("repair controller result exceeds its bounded size")

    @property
    def result_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def authorizes_effect(self) -> bool:
        return False

    @property
    def authorizes_merge(self) -> bool:
        return False

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REPAIR_CONTROLLER_RESULT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "disposition": self.disposition.value,
            "merge_disposition": self.merge_disposition.value,
            "selected_tier": None if self.selected_tier is None else self.selected_tier.value,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "plan_id": "" if self.plan is None else self.plan.plan_id,
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "receipt_id": "" if self.receipt is None else self.receipt.receipt_id,
            "reason_codes": list(self.reason_codes),
            "model_call_count": self.model_call_count,
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "satisfied_merge_conditions": list(self.satisfied_merge_conditions),
            "missing_merge_conditions": list(self.missing_merge_conditions),
            "source_edit_admitted": self.source_edit_admitted,
            "engine_report_id": self.engine_report_id,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "failure_signature": self.failure_signature,
            "authorizes_effect": False,
            "authorizes_merge": False,
            "low_risk_merge_conditions": list(LOW_RISK_MERGE_CONDITIONS),
        }
        if include_identity:
            payload["result_id"] = self.result_id
        return payload


class AutonomousRepairController:
    """Tier-selection and scope facade over ``AutonomousRepairEngine``.

    The controller never writes repository bytes, never mints merge
    authority, and never constructs a second repair engine.  When an engine
    is injected it must be the existing ``AutonomousRepairEngine``.
    """

    INTERFACE: Final[str] = AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    def __init__(
        self,
        *,
        engine: AutonomousRepairEngine | None = None,
        identical_failure_limit: int = DEFAULT_IDENTICAL_FAILURE_LIMIT,
        base_backoff_milliseconds: int = DEFAULT_BASE_BACKOFF_MS,
        max_backoff_milliseconds: int = DEFAULT_MAX_BACKOFF_MS,
    ) -> None:
        if engine is not None and not isinstance(engine, AutonomousRepairEngine):
            raise RepairControllerError("engine must be the existing AutonomousRepairEngine")
        self._engine = engine
        self._identical_failure_limit = _int(
            identical_failure_limit, "identical_failure_limit", minimum=1, maximum=1_024
        )
        self._base_backoff_milliseconds = _int(
            base_backoff_milliseconds, "base_backoff_milliseconds", minimum=1
        )
        self._max_backoff_milliseconds = _int(
            max_backoff_milliseconds, "max_backoff_milliseconds", minimum=1
        )
        if self._base_backoff_milliseconds > self._max_backoff_milliseconds:
            raise RepairControllerError("base backoff cannot exceed max backoff")
        self._failures: dict[str, RepairFailureRecord] = {}

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine(self) -> AutonomousRepairEngine | None:
        return self._engine

    @property
    def engine_authority(self) -> str:
        return "AutonomousRepairEngine@1"

    def snapshot(self) -> Mapping[str, Any]:
        payload = {
            "schema": REPAIR_FAILURE_MEMORY_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "identical_failure_limit": self._identical_failure_limit,
            "base_backoff_milliseconds": self._base_backoff_milliseconds,
            "max_backoff_milliseconds": self._max_backoff_milliseconds,
            "records": [record.to_dict() for record in self._failures.values()],
        }
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > MAX_REPAIR_CONTROLLER_SNAPSHOT_BYTES:
            raise RepairControllerError("repair failure snapshot exceeds its bounded size")
        return MappingProxyType(payload)

    def failure_record(self, signature: str) -> RepairFailureRecord | None:
        return self._failures.get(signature)

    def select_tier(self, attempt: AutonomousRepairAttempt | Mapping[str, Any]) -> RepairTier:
        return select_repair_tier(AutonomousRepairAttempt.from_mapping(attempt))

    def admit_source_edit(
        self,
        attempt: AutonomousRepairAttempt | Mapping[str, Any],
    ) -> AdmittedSourceEditOperator:
        bound = AutonomousRepairAttempt.from_mapping(attempt)
        return admit_source_edit_operator(
            bound.source_edit_operator,
            predicted_files=bound.predicted_files,
            allowed_paths=bound.allowed_paths,
            envelope_paths=bound.envelope.allowed_paths,
        )

    def _backoff_ms(self, count: int) -> int:
        shift = max(0, count - 1)
        value = self._base_backoff_milliseconds * (2**shift)
        return min(value, self._max_backoff_milliseconds)

    def _observe_identical_failure(
        self,
        attempt: AutonomousRepairAttempt,
        *,
        model_calls_this_attempt: int,
    ) -> RepairFailureRecord | None:
        signature = attempt.failure_signature
        if not signature:
            return None
        previous = self._failures.get(signature)
        diagnostic = attempt.diagnostic_receipt_id or (
            previous.diagnostic_receipt_id if previous is not None else ""
        )
        if previous is None:
            record = RepairFailureRecord(
                failure_signature=signature,
                diagnostic_receipt_id=diagnostic,
                count=1,
                backoff_milliseconds=0,
                model_call_count=model_calls_this_attempt,
            )
            self._failures[signature] = record
            return None
        count = previous.count + 1
        exhausted = count > self._identical_failure_limit
        record = RepairFailureRecord(
            failure_signature=signature,
            diagnostic_receipt_id=previous.diagnostic_receipt_id or diagnostic,
            count=count,
            backoff_milliseconds=self._backoff_ms(count - 1),
            model_call_count=previous.model_call_count,
            exhausted=exhausted,
        )
        self._failures[signature] = record
        return record

    def _build_plan(
        self,
        attempt: AutonomousRepairAttempt,
        tier: RepairTier,
    ) -> AutonomousRepairPlan:
        try:
            return AutonomousRepairPlan(
                objective_id=attempt.envelope.objective_id,
                task_id=attempt.envelope.task_id,
                repair_tier=tier,
                predicted_files=attempt.predicted_files,
                predicted_symbols=attempt.predicted_symbols,
                patch_envelope_id=attempt.patch_envelope_id,
                context_reference_ids=attempt.context_reference_ids,
                required_test_ids=attempt.required_test_ids,
                required_proof_ids=attempt.required_proof_ids,
                worktree_id=attempt.worktree_id,
                allowed_paths=attempt.allowed_paths,
                forbidden_symbols=attempt.forbidden_symbols,
                rollback_plan_id=attempt.rollback_plan_id,
                risk_class=attempt.envelope.risk_assessment.risk_class,
                max_changed_files=attempt.max_changed_files,
                max_changed_lines=attempt.max_changed_lines,
            )
        except AutonomyContractError as exc:
            raise RepairControllerError(_as_reason(str(exc))) from exc

    def _build_receipt(
        self,
        attempt: AutonomousRepairAttempt,
        plan: AutonomousRepairPlan | None,
        *,
        terminal_status: TerminalStatus,
        failure_signature: str = "",
        diagnostic_receipt_id: str = "",
    ) -> AutonomousRepairReceipt | None:
        if plan is None:
            return None
        try:
            return AutonomousRepairReceipt(
                plan_id=plan.plan_id,
                envelope_id=attempt.envelope.envelope_id,
                terminal_status=terminal_status,
                changed_paths=attempt.changed_paths,
                validation_receipt_ids=attempt.validation_receipt_ids,
                proof_receipt_ids=attempt.proof_receipt_ids,
                adversarial_assurance_receipt_ids=attempt.adversarial_assurance_receipt_ids,
                rollback_receipt_id=(
                    attempt.rollback_plan_id
                    if terminal_status
                    in {TerminalStatus.FAILED, TerminalStatus.BLOCKED, TerminalStatus.EXHAUSTED}
                    else ""
                ),
                failure_signature=failure_signature,
                diagnostic_receipt_id=diagnostic_receipt_id,
                authorizes_merge=False,
            )
        except AutonomyContractError as exc:
            raise RepairControllerError(_as_reason(str(exc))) from exc

    def _maybe_run_engine(
        self,
        attempt: AutonomousRepairAttempt,
        tier: RepairTier,
        *,
        delegate_to_engine: bool,
    ) -> str:
        if (
            not delegate_to_engine
            or self._engine is None
            or tier is RepairTier.MODEL_ASSISTED_BOUNDED
        ):
            return ""
        # Analysis-only delegation.  The engine never counts as completion.
        try:
            report = self._engine.run(
                [
                    {
                        "work_id": f"work:{attempt.envelope.task_id}",
                        "operation": attempt.predicted_symbols[0],
                        "path": attempt.predicted_files[0],
                        "symbol": attempt.predicted_symbols[0],
                        "write_paths": list(attempt.predicted_files),
                        "domain": "agent_supervisor",
                    }
                ]
            )
        except Exception as exc:  # noqa: BLE001 - facade must fail closed
            raise RepairControllerError("repair_engine_delegation_failed") from exc
        if getattr(report, "passed", False):
            raise RepairControllerError("repair_engine_cannot_authorize_completion")
        payload = {
            "delegated_engine": "AutonomousRepairEngine@1",
            "passed": False,
            "model_call_count": int(getattr(report, "model_call_count", 0) or 0),
            "row_count": len(getattr(report, "rows", ()) or ()),
        }
        return content_identity(payload)

    def _invoke_model(
        self,
        plan: AutonomousRepairPlan,
        attempt: AutonomousRepairAttempt,
        model_call: ModelAssistedRepairHook | None,
    ) -> int:
        if model_call is None:
            return 0
        model_call(plan, attempt)
        return 1

    def repair(
        self,
        attempt: AutonomousRepairAttempt | Mapping[str, Any],
        *,
        model_call: ModelAssistedRepairHook | None = None,
        delegate_to_engine: bool = False,
    ) -> AutonomousRepairControllerResult:
        """Admit, optionally delegate, and emit merge eligibility.

        The optional ``model_call`` hook is invoked at most once per distinct
        failure signature.  Identical failures reuse the stored diagnostic.
        """

        bound = AutonomousRepairAttempt.from_mapping(attempt)
        rejection = _scope_reasons(bound)
        source_edit_admitted = False
        if bound.source_edit_operator is not None and not rejection:
            try:
                self.admit_source_edit(bound)
                source_edit_admitted = True
            except (AdmittedSourceEditError, RepairControllerError) as exc:
                rejection = (*rejection, _as_reason(str(exc) or "source_edit_not_admitted"))

        tier = select_repair_tier(bound)
        rejection = (*rejection, *_plan_construction_gaps(bound, tier))

        identical = None
        if bound.failure_signature and not rejection:
            previous = self._failures.get(bound.failure_signature)
            if previous is not None:
                identical = self._observe_identical_failure(bound, model_calls_this_attempt=0)
                if identical is None:
                    raise RepairControllerError("identical_failure_record_missing")
                disposition = (
                    RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
                    if identical.exhausted
                    else RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
                )
                terminal = (
                    TerminalStatus.EXHAUSTED
                    if identical.exhausted
                    else TerminalStatus.BLOCKED
                )
                plan = None
                try:
                    plan = self._build_plan(bound, tier)
                except RepairControllerError:
                    plan = None
                receipt = self._build_receipt(
                    bound,
                    plan,
                    terminal_status=terminal,
                    failure_signature=bound.failure_signature,
                    diagnostic_receipt_id=identical.diagnostic_receipt_id,
                )
                satisfied, missing = evaluate_low_risk_merge_conditions(
                    bound, terminal_status=terminal, plan=plan
                )
                return AutonomousRepairControllerResult(
                    disposition=disposition,
                    merge_disposition=RepairMergeDisposition.NONE,
                    selected_tier=tier,
                    plan=plan,
                    receipt=receipt,
                    reason_codes=(
                        disposition.value,
                        "identical_failure_reused_diagnostic",
                        "model_call_suppressed",
                    ),
                    model_call_count=0,
                    diagnostic_reused=True,
                    backoff_milliseconds=identical.backoff_milliseconds,
                    satisfied_merge_conditions=satisfied,
                    missing_merge_conditions=missing,
                    source_edit_admitted=source_edit_admitted,
                    diagnostic_receipt_id=identical.diagnostic_receipt_id,
                    failure_signature=bound.failure_signature,
                )

        if rejection:
            terminal = TerminalStatus.BLOCKED
            plan = None
            receipt = None
            satisfied, missing = evaluate_low_risk_merge_conditions(
                bound, terminal_status=terminal, plan=None
            )
            return AutonomousRepairControllerResult(
                disposition=RepairControllerDisposition.REJECTED,
                merge_disposition=RepairMergeDisposition.REJECTED,
                selected_tier=tier,
                plan=plan,
                receipt=receipt,
                reason_codes=rejection,
                model_call_count=0,
                satisfied_merge_conditions=satisfied,
                missing_merge_conditions=missing,
                source_edit_admitted=False,
                failure_signature=bound.failure_signature,
            )

        try:
            plan = self._build_plan(bound, tier)
        except RepairControllerError as exc:
            satisfied, missing = evaluate_low_risk_merge_conditions(
                bound, terminal_status=TerminalStatus.BLOCKED, plan=None
            )
            return AutonomousRepairControllerResult(
                disposition=RepairControllerDisposition.REJECTED,
                merge_disposition=RepairMergeDisposition.REJECTED,
                selected_tier=tier,
                reason_codes=(*rejection, _as_reason(str(exc))),
                satisfied_merge_conditions=satisfied,
                missing_merge_conditions=missing,
                failure_signature=bound.failure_signature,
            )
        try:
            engine_report_id = self._maybe_run_engine(
                bound, tier, delegate_to_engine=delegate_to_engine
            )
        except RepairControllerError as exc:
            satisfied, missing = evaluate_low_risk_merge_conditions(
                bound, terminal_status=TerminalStatus.BLOCKED, plan=plan
            )
            return AutonomousRepairControllerResult(
                disposition=RepairControllerDisposition.REJECTED,
                merge_disposition=RepairMergeDisposition.REJECTED,
                selected_tier=tier,
                plan=plan,
                reason_codes=(_as_reason(str(exc)),),
                satisfied_merge_conditions=satisfied,
                missing_merge_conditions=missing,
                failure_signature=bound.failure_signature,
            )
        model_calls = 0
        if tier is RepairTier.MODEL_ASSISTED_BOUNDED:
            model_calls = self._invoke_model(plan, bound, model_call)

        if bound.validation_receipt_ids:
            terminal = TerminalStatus.SUCCEEDED
        else:
            terminal = TerminalStatus.PENDING

        if bound.failure_signature and terminal is not TerminalStatus.SUCCEEDED:
            self._observe_identical_failure(bound, model_calls_this_attempt=model_calls)

        receipt = self._build_receipt(
            bound,
            plan,
            terminal_status=terminal,
            failure_signature=bound.failure_signature,
            diagnostic_receipt_id=bound.diagnostic_receipt_id,
        )
        satisfied, missing = evaluate_low_risk_merge_conditions(
            bound, terminal_status=terminal, plan=plan
        )
        merge = _merge_disposition_for(
            bound,
            terminal_status=terminal,
            rejection_reasons=(),
            satisfied=satisfied,
            missing=missing,
        )
        if merge is RepairMergeDisposition.AUTONOMOUS_MERGE:
            disposition = RepairControllerDisposition.AUTONOMOUS_MERGE_ELIGIBLE
        elif merge is RepairMergeDisposition.PROPOSAL:
            disposition = RepairControllerDisposition.PROPOSAL
        elif merge is RepairMergeDisposition.ROLLBACK:
            disposition = RepairControllerDisposition.ROLLBACK
        elif terminal is TerminalStatus.SUCCEEDED:
            disposition = RepairControllerDisposition.EXECUTED
        else:
            disposition = RepairControllerDisposition.ADMITTED
        reasons = [disposition.value, f"tier:{tier.value}"]
        if source_edit_admitted:
            reasons.append("source_edit_admitted")
        if engine_report_id:
            reasons.append("delegated_existing_repair_engine")
        if model_calls:
            reasons.append("model_assisted_invoked")
        if merge is RepairMergeDisposition.AUTONOMOUS_MERGE:
            reasons.append("low_risk_merge_conjunction")
        if merge is RepairMergeDisposition.PROPOSAL and (
            bound.envelope.risk_assessment.risk_class
            is RiskClass.R3_BOUNDED_REPOSITORY_MUTATION
        ):
            reasons.append("r3_proposal")
        return AutonomousRepairControllerResult(
            disposition=disposition,
            merge_disposition=merge,
            selected_tier=tier,
            plan=plan,
            receipt=receipt,
            reason_codes=tuple(reasons),
            model_call_count=model_calls,
            diagnostic_reused=False,
            backoff_milliseconds=0,
            satisfied_merge_conditions=satisfied,
            missing_merge_conditions=missing,
            source_edit_admitted=source_edit_admitted,
            engine_report_id=engine_report_id,
            diagnostic_receipt_id=bound.diagnostic_receipt_id,
            failure_signature=bound.failure_signature,
        )


__all__ = [
    "AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE",
    "DEFAULT_FORBIDDEN_SYMBOLS",
    "LOW_RISK_MERGE_CONDITIONS",
    "PROTECTED_AUTHORITY_PATHS",
    "SELF_EDIT_PATHS",
    "VALIDATOR_POLICY_KEY_PATHS",
    "AdmittedSourceEditError",
    "AutonomousRepairAttempt",
    "AutonomousRepairController",
    "AutonomousRepairControllerResult",
    "RepairControllerDisposition",
    "RepairControllerError",
    "RepairFailureRecord",
    "RepairMergeDisposition",
    "admit_source_edit_operator",
    "evaluate_low_risk_merge_conditions",
    "is_protected_authority_path",
    "is_self_edit_path",
    "is_validator_policy_key_path",
    "path_is_within",
    "select_repair_tier",
]
