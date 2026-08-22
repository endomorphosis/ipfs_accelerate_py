# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded facade over the existing autonomous-repair engine.

``AutonomousRepairController@1`` selects a deterministic, template-constrained,
or model-assisted repair tier and binds one ``AutonomyEnvelope``.  It is not a
second repair engine, materializer, merge authority, or ``DecisionRuntime``.

Admission and execution stay with
``agent_supervisor.autonomous_repair.engine.AutonomousRepairEngine`` and the
existing source-edit operator.  This module rejects scope escape, self-edit,
and validator/policy/key mutation; reuses diagnosis for identical failures;
and treats autonomous merge as the conjunction of every stated low-risk
condition.  Repair receipts remain evidence and never authorize merge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..autonomous_repair.contracts import AUTONOMOUS_REPAIR_INTERFACE
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
REPAIR_CONTROLLER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-result@1"
)
REPAIR_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-request@1"
)
SOURCE_EDIT_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/source-edit-admission@1"
)
REPAIR_CONTROLLER_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-snapshot@1"
)
REPAIR_FAILURE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-failure-record@1"
)

CANONICAL_REPAIR_ENGINE_TYPE: Final[type] = AutonomousRepairEngine
CANONICAL_REPAIR_ENGINE_INTERFACE: Final[str] = AUTONOMOUS_REPAIR_INTERFACE
CANONICAL_SOURCE_EDIT_OPERATOR_TYPE: Final[type] = AdmittedSourceEditOperator

CONTROLLER_RELATIVE_PATH: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
)

SELF_PROTECTING_PATHS: Final[tuple[str, ...]] = (
    CONTROLLER_RELATIVE_PATH,
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair/engine.py",
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair/no_llm_policy.py",
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair/materialize.py",
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair/edit_plan.py",
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair/validation.py",
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair/publish.py",
    "ipfs_accelerate_py/agent_supervisor/context/decision_runtime.py",
)

PROTECTED_AUTHORITY_PATHS: Final[tuple[str, ...]] = (
    ".gitignore",
    "docs/architecture/AGENT_SUPERVISOR_AUTONOMOUS_META_CONTROLLER_PLAN.md",
    "docs/architecture/agent_supervisor_autonomous_meta_controller.objectives.md",
    "docs/architecture/agent_supervisor_autonomous_meta_controller.todo.md",
    "config/agent_supervisor_autonomous_meta_controller_scheduler.json",
    "scripts/validate_agent_supervisor_autonomous_meta_controller_board.py",
    "scripts/materialize_agent_supervisor_autonomous_meta_controller_board.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/llm_router.py",
    *SELF_PROTECTING_PATHS,
)

_VALIDATOR_POLICY_KEY_SEGMENTS: Final[frozenset[str]] = frozenset(
    {
        "credentials",
        "oracle",
        "oracles",
        "policies",
        "policy",
        "private_keys",
        "secrets",
        "signing_keys",
        "trusted_keys",
        "verification",
        "verifier",
        "verifiers",
    }
)
_VALIDATOR_POLICY_KEY_SUFFIXES: Final[tuple[str, ...]] = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
)
_VALIDATOR_POLICY_KEY_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        "authorized_keys",
        "benchmark_oracle.json",
        "golden.json",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "oracle.json",
        "policy.json",
        "policy.yaml",
        "policy.yml",
    }
)
_VALIDATOR_POLICY_KEY_PREFIXES: Final[tuple[str, ...]] = (
    ".ssh/",
    ".aws/",
    ".gnupg/",
    "config/",
    "secrets/",
    "credentials/",
    "ipfs_accelerate_py/agent_supervisor/verification/",
    "ipfs_accelerate_py/agent_supervisor/proof/",
    "ipfs_accelerate_py/agent_supervisor/validation/",
)

DEFAULT_FORBIDDEN_SYMBOLS: Final[tuple[str, ...]] = (
    "authorized_keys",
    "signing_key",
    "trusted_keys",
    "validator_policy",
)

LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "policy_autonomous_merge_enabled",
    "risk_class_r2_or_lower",
    "reversible",
    "autonomy_level_admits_execution",
    "scope_within_envelope",
    "no_scope_escape",
    "no_self_edit",
    "no_validator_policy_key_mutation",
    "exact_predicted_files",
    "exact_predicted_symbols",
    "predicted_files_within_patch_bounds",
    "required_tests_satisfied",
    "required_proofs_satisfied",
    "validation_receipts_current",
    "isolated_worktree_bound",
    "rollback_plan_bound",
    "terminal_succeeded",
    "repair_receipt_does_not_authorize_merge",
    "not_identical_failure_backoff",
    "not_r3_or_higher",
    "changed_paths_within_envelope",
    "changed_paths_subset_of_predicted",
)

_MUTATING_LEVELS: Final[frozenset[AutonomyLevel]] = frozenset(
    {
        AutonomyLevel.EXECUTE_REVERSIBLE,
        AutonomyLevel.EXECUTE_BOUNDED_MUTATION,
        AutonomyLevel.SELF_REPAIR_ISOLATED,
    }
)
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

MAX_REPAIR_CONTROLLER_SNAPSHOT_BYTES: Final[int] = 4 * MAX_CANONICAL_RECORD_BYTES
DEFAULT_BASE_BACKOFF_MS: Final[int] = 100
DEFAULT_MAX_BACKOFF_MS: Final[int] = 10_000
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 3
DEFAULT_MAX_CHANGED_FILES: Final[int] = 8
DEFAULT_MAX_CHANGED_LINES: Final[int] = 400


class RepairControllerError(ValueError):
    """Raised when repair-controller inputs violate the frozen facade contract."""


class RepairControllerDisposition(str, Enum):
    """Closed outcome of one facade step.  None grants merge or effect authority."""

    ADMITTED = "admitted"
    EXECUTED = "executed"
    IDENTICAL_FAILURE_BACKOFF = "identical_failure_backoff"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    REJECTED_SCOPE_ESCAPE = "rejected_scope_escape"
    REJECTED_SELF_EDIT = "rejected_self_edit"
    REJECTED_VALIDATOR_POLICY_KEY = "rejected_validator_policy_key"
    REJECTED_MISSING_PRECONDITIONS = "rejected_missing_preconditions"
    REJECTED_UNADMITTED_SOURCE_EDIT = "rejected_unadmitted_source_edit"
    REJECTED_SECOND_ENGINE = "rejected_second_engine"
    ROLLBACK_REQUIRED = "rollback_required"
    REPAIR_BOUND_EXCEEDED = "repair_bound_exceeded"


class RepairMergeDisposition(str, Enum):
    """Merge recommendation only.  Existing merge authorities remain canonical."""

    HOLD = "hold"
    AUTONOMOUS_MERGE_ELIGIBLE = "autonomous_merge_eligible"
    PROPOSAL_REQUIRED = "proposal_required"
    REJECTED = "rejected"
    ROLLBACK = "rollback"


ModelInvoker = Callable[["RepairRequest"], Mapping[str, Any]]


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
    preserve_order: bool = False,
) -> tuple[str, ...]:
    raw = _identifiers(value, name, required=required, preserve_order=True)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        path = _posix_path(item, name)
        if path not in seen:
            seen.add(path)
            normalized.append(path)
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


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise RepairControllerError(
            f"{name} must be an integer between {minimum} and {maximum}"
        )
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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise RepairControllerError(f"{name} must be an object")
    if len(value) > MAX_MAPPING_ITEMS:
        raise RepairControllerError(f"{name} contains too many items")
    _reject_forbidden_keys(value, name)
    return MappingProxyType(dict(value))


def path_under_prefix(path: str, prefix: str) -> bool:
    """Return True when *path* equals *prefix* or is a descendant of it."""

    path_n = path.replace("\\", "/").strip()
    pref = prefix.replace("\\", "/").strip()
    if not pref or not path_n:
        return False
    if pref.endswith("/"):
        pref = pref.rstrip("/")
    return path_n == pref or path_n.startswith(pref + "/")


def path_under_any(path: str, prefixes: Sequence[str]) -> bool:
    return any(path_under_prefix(path, prefix) for prefix in prefixes)


def is_self_edit_path(path: str) -> bool:
    """Return True when *path* would rewrite this facade or its engine."""

    try:
        normalized = _posix_path(path, "path")
    except RepairControllerError:
        return True
    return path_under_any(normalized, SELF_PROTECTING_PATHS)


def is_validator_policy_key_path(path: str) -> bool:
    """Return True when *path* is a validator, policy, or trusted-key surface."""

    try:
        normalized = _posix_path(path, "path")
    except RepairControllerError:
        return True
    lower = normalized.lower()
    if path_under_any(lower, _VALIDATOR_POLICY_KEY_PREFIXES):
        return True
    if path_under_any(normalized, PROTECTED_AUTHORITY_PATHS) and not is_self_edit_path(
        normalized
    ):
        return True
    if PurePosixPath(lower).name in _VALIDATOR_POLICY_KEY_BASENAMES:
        return True
    if any(lower.endswith(suffix) for suffix in _VALIDATOR_POLICY_KEY_SUFFIXES):
        return True
    return any(part in _VALIDATOR_POLICY_KEY_SEGMENTS for part in PurePosixPath(lower).parts)


def classify_forbidden_paths(paths: Sequence[str], allowed_paths: Sequence[str]) -> tuple[str, ...]:
    """Return closed reason codes for paths that cannot be repaired."""

    reasons: list[str] = []
    for path in paths:
        if is_self_edit_path(path):
            reasons.append("self_edit")
        elif is_validator_policy_key_path(path):
            reasons.append("validator_policy_key")
        elif allowed_paths and not path_under_any(path, allowed_paths):
            reasons.append("scope_escape")
    ordered: list[str] = []
    for item in ("self_edit", "validator_policy_key", "scope_escape"):
        if item in reasons and item not in ordered:
            ordered.append(item)
    return tuple(ordered)


def _backoff_milliseconds(failure_count: int, *, base: int, maximum: int) -> int:
    if failure_count <= 0:
        return 0
    shift = min(failure_count - 1, 16)
    return min(maximum, base * (2**shift))


@dataclass(frozen=True)
class RepairBackoffPolicy:
    """Integer backoff bounds for identical failure signatures."""

    base_backoff_milliseconds: int = DEFAULT_BASE_BACKOFF_MS
    max_backoff_milliseconds: int = DEFAULT_MAX_BACKOFF_MS
    max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_backoff_milliseconds",
            _int(self.base_backoff_milliseconds, "base_backoff_milliseconds", minimum=1),
        )
        object.__setattr__(
            self,
            "max_backoff_milliseconds",
            _int(self.max_backoff_milliseconds, "max_backoff_milliseconds", minimum=1),
        )
        object.__setattr__(
            self,
            "max_identical_failures",
            _int(self.max_identical_failures, "max_identical_failures", minimum=1),
        )
        if self.base_backoff_milliseconds > self.max_backoff_milliseconds:
            raise RepairControllerError("base backoff cannot exceed max backoff")


@dataclass(frozen=True)
class RepairFailureRecord:
    """Compact identical-failure memory.  Bodies and prompts are forbidden."""

    failure_signature: str
    diagnostic_receipt_id: str
    count: int
    model_call_count: int
    last_backoff_milliseconds: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failure_signature",
            _identifier(self.failure_signature, "failure_signature"),
        )
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _identifier(self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False),
        )
        object.__setattr__(self, "count", _int(self.count, "count", minimum=1))
        object.__setattr__(
            self, "model_call_count", _int(self.model_call_count, "model_call_count")
        )
        object.__setattr__(
            self,
            "last_backoff_milliseconds",
            _int(self.last_backoff_milliseconds, "last_backoff_milliseconds"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_FAILURE_RECORD_SCHEMA,
            "failure_signature": self.failure_signature,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "count": self.count,
            "model_call_count": self.model_call_count,
            "last_backoff_milliseconds": self.last_backoff_milliseconds,
        }


@dataclass(frozen=True)
class RepairRequest:
    """One bounded repair attempt.  It never carries source bodies or prompts."""

    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    context_reference_ids: tuple[str, ...]
    worktree_id: str
    rollback_plan_id: str
    required_test_ids: tuple[str, ...] = ()
    required_proof_ids: tuple[str, ...] = ()
    completed_test_ids: tuple[str, ...] = ()
    completed_proof_ids: tuple[str, ...] = ()
    forbidden_symbols: tuple[str, ...] = DEFAULT_FORBIDDEN_SYMBOLS
    requested_tier: RepairTier | None = None
    template_id: str = ""
    requires_model: bool = False
    max_changed_files: int = DEFAULT_MAX_CHANGED_FILES
    max_changed_lines: int = DEFAULT_MAX_CHANGED_LINES
    changed_line_count: int = 0
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""
    changed_paths: tuple[str, ...] = ()
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    terminal_status: TerminalStatus = TerminalStatus.PENDING
    source_edit_operator: Mapping[str, Any] | None = None
    work_items: tuple[Mapping[str, Any], ...] = ()
    allow_code_edit_materialize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_files",
            _posix_paths(self.predicted_files, "predicted_files", required=True),
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _identifiers(self.predicted_symbols, "predicted_symbols", required=True),
        )
        object.__setattr__(
            self,
            "context_reference_ids",
            _identifiers(
                self.context_reference_ids, "context_reference_ids", required=True
            ),
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
            "required_test_ids",
            "required_proof_ids",
            "completed_test_ids",
            "completed_proof_ids",
            "validation_receipt_ids",
            "proof_receipt_ids",
            "adversarial_assurance_receipt_ids",
        ):
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
        object.__setattr__(
            self,
            "forbidden_symbols",
            _identifiers(self.forbidden_symbols, "forbidden_symbols"),
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
            self, "requires_model", _bool(self.requires_model, "requires_model")
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
        object.__setattr__(
            self, "changed_line_count", _int(self.changed_line_count, "changed_line_count")
        )
        object.__setattr__(
            self,
            "failure_signature",
            _identifier(self.failure_signature, "failure_signature", required=False),
        )
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _identifier(
                self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False
            ),
        )
        object.__setattr__(
            self, "changed_paths", _posix_paths(self.changed_paths, "changed_paths")
        )
        object.__setattr__(
            self,
            "terminal_status",
            _enum(self.terminal_status, TerminalStatus, "terminal_status"),
        )
        operator = self.source_edit_operator
        if operator is None:
            object.__setattr__(self, "source_edit_operator", None)
        elif isinstance(operator, AdmittedSourceEditOperator):
            object.__setattr__(
                self,
                "source_edit_operator",
                MappingProxyType(
                    {
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
                ),
            )
        else:
            object.__setattr__(self, "source_edit_operator", _mapping(operator, "source_edit_operator"))
        items = self.work_items
        if items is None:
            coerced_items: tuple[Mapping[str, Any], ...] = ()
        elif isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
            if len(items) > MAX_SEQUENCE_ITEMS:
                raise RepairControllerError("work_items contains too many items")
            coerced_items = tuple(_mapping(item, "work_items") for item in items)
        else:
            raise RepairControllerError("work_items must be a sequence of objects")
        object.__setattr__(self, "work_items", coerced_items)
        object.__setattr__(
            self,
            "allow_code_edit_materialize",
            _bool(self.allow_code_edit_materialize, "allow_code_edit_materialize"),
        )
        if len(self.predicted_files) > self.max_changed_files:
            raise RepairControllerError("predicted repair files exceed the patch envelope")
        if self.changed_line_count > self.max_changed_lines:
            raise RepairControllerError("changed line count exceeds the patch envelope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_REQUEST_SCHEMA,
            "predicted_files": list(self.predicted_files),
            "predicted_symbols": list(self.predicted_symbols),
            "context_reference_ids": list(self.context_reference_ids),
            "worktree_id": self.worktree_id,
            "rollback_plan_id": self.rollback_plan_id,
            "required_test_ids": list(self.required_test_ids),
            "required_proof_ids": list(self.required_proof_ids),
            "completed_test_ids": list(self.completed_test_ids),
            "completed_proof_ids": list(self.completed_proof_ids),
            "forbidden_symbols": list(self.forbidden_symbols),
            "requested_tier": None if self.requested_tier is None else self.requested_tier.value,
            "template_id": self.template_id,
            "requires_model": self.requires_model,
            "max_changed_files": self.max_changed_files,
            "max_changed_lines": self.max_changed_lines,
            "changed_line_count": self.changed_line_count,
            "failure_signature": self.failure_signature,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "changed_paths": list(self.changed_paths),
            "validation_receipt_ids": list(self.validation_receipt_ids),
            "proof_receipt_ids": list(self.proof_receipt_ids),
            "adversarial_assurance_receipt_ids": list(self.adversarial_assurance_receipt_ids),
            "terminal_status": self.terminal_status.value,
            "source_edit_operator": (
                None if self.source_edit_operator is None else dict(self.source_edit_operator)
            ),
            "work_items": [dict(item) for item in self.work_items],
            "allow_code_edit_materialize": self.allow_code_edit_materialize,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | RepairRequest) -> RepairRequest:
        if isinstance(payload, RepairRequest):
            return payload
        if not isinstance(payload, Mapping):
            raise RepairControllerError("repair request must be an object")
        _reject_forbidden_keys(payload, "repair request")
        if payload.get("schema") not in (None, "", REPAIR_REQUEST_SCHEMA):
            raise RepairControllerError("unsupported repair request schema")
        return cls(
            predicted_files=tuple(payload.get("predicted_files") or ()),
            predicted_symbols=tuple(payload.get("predicted_symbols") or ()),
            context_reference_ids=tuple(payload.get("context_reference_ids") or ()),
            worktree_id=payload.get("worktree_id", ""),
            rollback_plan_id=payload.get("rollback_plan_id", ""),
            required_test_ids=tuple(payload.get("required_test_ids") or ()),
            required_proof_ids=tuple(payload.get("required_proof_ids") or ()),
            completed_test_ids=tuple(payload.get("completed_test_ids") or ()),
            completed_proof_ids=tuple(payload.get("completed_proof_ids") or ()),
            forbidden_symbols=tuple(
                payload.get("forbidden_symbols")
                if "forbidden_symbols" in payload
                else DEFAULT_FORBIDDEN_SYMBOLS
            ),
            requested_tier=payload.get("requested_tier"),
            template_id=str(payload.get("template_id") or ""),
            requires_model=payload.get("requires_model", False),
            max_changed_files=payload.get("max_changed_files", DEFAULT_MAX_CHANGED_FILES),
            max_changed_lines=payload.get("max_changed_lines", DEFAULT_MAX_CHANGED_LINES),
            changed_line_count=payload.get("changed_line_count", 0),
            failure_signature=str(payload.get("failure_signature") or ""),
            diagnostic_receipt_id=str(payload.get("diagnostic_receipt_id") or ""),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            validation_receipt_ids=tuple(payload.get("validation_receipt_ids") or ()),
            proof_receipt_ids=tuple(payload.get("proof_receipt_ids") or ()),
            adversarial_assurance_receipt_ids=tuple(
                payload.get("adversarial_assurance_receipt_ids") or ()
            ),
            terminal_status=payload.get("terminal_status", TerminalStatus.PENDING),
            source_edit_operator=payload.get("source_edit_operator"),
            work_items=tuple(payload.get("work_items") or ()),
            allow_code_edit_materialize=payload.get("allow_code_edit_materialize", False),
        )


@dataclass(frozen=True)
class SourceEditAdmission:
    """Admission of one exact source-edit operator.  Application is separate."""

    admitted: bool
    disposition: str
    reasons: tuple[str, ...]
    operator_id: str = ""
    relative_path: str = ""
    old_digest: str = ""
    new_digest: str = ""
    mutation_applied: bool = False
    completion_authoritative: bool = False
    grants_execution_authority: bool = False
    authorizes_merge: bool = False
    policy_flag_is_not_source_edit_admission: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(
            self, "disposition", _identifier(self.disposition, "disposition")
        )
        object.__setattr__(
            self,
            "reasons",
            _identifiers(self.reasons, "reasons", preserve_order=True),
        )
        object.__setattr__(
            self, "operator_id", _identifier(self.operator_id, "operator_id", required=False)
        )
        object.__setattr__(
            self,
            "relative_path",
            _identifier(self.relative_path, "relative_path", required=False),
        )
        if self.relative_path:
            object.__setattr__(
                self, "relative_path", _posix_path(self.relative_path, "relative_path")
            )
        object.__setattr__(
            self, "old_digest", _identifier(self.old_digest, "old_digest", required=False)
        )
        object.__setattr__(
            self, "new_digest", _identifier(self.new_digest, "new_digest", required=False)
        )
        for name in (
            "mutation_applied",
            "completion_authoritative",
            "grants_execution_authority",
            "authorizes_merge",
            "policy_flag_is_not_source_edit_admission",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if self.mutation_applied or self.completion_authoritative:
            raise RepairControllerError("source-edit admission cannot apply or complete a repair")
        if self.grants_execution_authority or self.authorizes_merge:
            raise RepairControllerError("source-edit admission cannot grant execution or merge")
        if self.admitted and self.disposition != "validation_pending":
            raise RepairControllerError("admitted source edits remain validation-pending")
        if not self.policy_flag_is_not_source_edit_admission:
            raise RepairControllerError("a policy flag is not source-edit admission")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_EDIT_ADMISSION_SCHEMA,
            "admitted": self.admitted,
            "disposition": self.disposition,
            "reasons": list(self.reasons),
            "operator_id": self.operator_id,
            "relative_path": self.relative_path,
            "old_digest": self.old_digest,
            "new_digest": self.new_digest,
            "mutation_applied": False,
            "completion_authoritative": False,
            "grants_execution_authority": False,
            "authorizes_merge": False,
            "policy_flag_is_not_source_edit_admission": True,
        }


@dataclass(frozen=True)
class RepairControllerResult:
    """Facade result.  ``authorizes_merge`` and ``authorizes_effect`` stay false."""

    disposition: RepairControllerDisposition
    merge_disposition: RepairMergeDisposition
    selected_tier: RepairTier | None = None
    plan: AutonomousRepairPlan | None = None
    receipt: AutonomousRepairReceipt | None = None
    reason_codes: tuple[str, ...] = ()
    model_call_count: int = 0
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    merge_condition_results: Mapping[str, bool] = field(default_factory=dict)
    engine_interface: str = CANONICAL_REPAIR_ENGINE_INTERFACE
    engine_report_passed: bool = False
    source_edit_admitted: bool = False
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""

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
        raw_conditions = _mapping(self.merge_condition_results, "merge_condition_results")
        if set(raw_conditions) != set(LOW_RISK_MERGE_CONDITIONS):
            raise RepairControllerError(
                "merge_condition_results must report every stated low-risk condition"
            )
        conditions = {
            name: _bool(raw_conditions[name], "merge_condition_results")
            for name in LOW_RISK_MERGE_CONDITIONS
        }
        object.__setattr__(self, "merge_condition_results", MappingProxyType(conditions))
        object.__setattr__(
            self,
            "engine_interface",
            _identifier(self.engine_interface, "engine_interface"),
        )
        if self.engine_interface != CANONICAL_REPAIR_ENGINE_INTERFACE:
            raise RepairControllerError("repair facade must name the existing engine interface")
        object.__setattr__(
            self,
            "engine_report_passed",
            _bool(self.engine_report_passed, "engine_report_passed"),
        )
        object.__setattr__(
            self,
            "source_edit_admitted",
            _bool(self.source_edit_admitted, "source_edit_admitted"),
        )
        object.__setattr__(
            self,
            "failure_signature",
            _identifier(self.failure_signature, "failure_signature", required=False),
        )
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _identifier(
                self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False
            ),
        )
        if self.receipt is not None and self.receipt.authorizes_merge:
            raise RepairControllerError("repair receipts cannot independently authorize merge")
        if (
            self.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
            and not all(conditions.values())
        ):
            raise RepairControllerError(
                "autonomous merge requires the conjunction of every low-risk condition"
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
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "reason_codes": list(self.reason_codes),
            "model_call_count": self.model_call_count,
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "merge_condition_results": dict(self.merge_condition_results),
            "engine_interface": self.engine_interface,
            "engine_report_passed": self.engine_report_passed,
            "source_edit_admitted": self.source_edit_admitted,
            "failure_signature": self.failure_signature,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "authorizes_effect": False,
            "authorizes_merge": False,
        }
        if include_identity:
            payload["result_id"] = self.result_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepairControllerResult:
        expected = {
            "schema",
            "interface",
            "program_id",
            "disposition",
            "merge_disposition",
            "selected_tier",
            "plan",
            "receipt",
            "reason_codes",
            "model_call_count",
            "diagnostic_reused",
            "backoff_milliseconds",
            "merge_condition_results",
            "engine_interface",
            "engine_report_passed",
            "source_edit_admitted",
            "failure_signature",
            "diagnostic_receipt_id",
            "authorizes_effect",
            "authorizes_merge",
            "result_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise RepairControllerError("repair controller result must use the closed schema")
        _reject_forbidden_keys(payload, "repair controller result")
        if payload.get("schema") != REPAIR_CONTROLLER_RESULT_SCHEMA:
            raise RepairControllerError("unsupported repair controller result schema")
        if payload.get("interface") != AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE:
            raise RepairControllerError("unsupported repair controller interface")
        if payload.get("program_id") != AUTONOMOUS_META_CONTROLLER_PROGRAM_ID:
            raise RepairControllerError("repair controller program identity is unsupported")
        if payload.get("authorizes_effect") is not False:
            raise RepairControllerError("repair controller results cannot authorize effects")
        if payload.get("authorizes_merge") is not False:
            raise RepairControllerError("repair controller results cannot authorize merge")
        plan_payload = payload.get("plan")
        receipt_payload = payload.get("receipt")
        result = cls(
            disposition=payload.get("disposition", ""),
            merge_disposition=payload.get("merge_disposition", ""),
            selected_tier=payload.get("selected_tier"),
            plan=None if plan_payload is None else AutonomousRepairPlan.from_dict(plan_payload),
            receipt=(
                None
                if receipt_payload is None
                else AutonomousRepairReceipt.from_dict(receipt_payload)
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            model_call_count=payload.get("model_call_count", -1),
            diagnostic_reused=payload.get("diagnostic_reused"),
            backoff_milliseconds=payload.get("backoff_milliseconds", -1),
            merge_condition_results=payload.get("merge_condition_results") or {},
            engine_interface=payload.get("engine_interface", ""),
            engine_report_passed=payload.get("engine_report_passed"),
            source_edit_admitted=payload.get("source_edit_admitted"),
            failure_signature=str(payload.get("failure_signature") or ""),
            diagnostic_receipt_id=str(payload.get("diagnostic_receipt_id") or ""),
        )
        if payload.get("result_id") != result.result_id:
            raise RepairControllerError("repair controller result identity does not match payload")
        return result


def evaluate_merge_conditions(
    *,
    policy: AutonomyPolicy,
    envelope: AutonomyEnvelope,
    request: RepairRequest,
    disposition: RepairControllerDisposition,
    receipt: AutonomousRepairReceipt | None,
    forbidden_reasons: Sequence[str],
) -> dict[str, bool]:
    """Evaluate every stated low-risk merge condition independently."""

    risk = envelope.risk_assessment.risk_class
    changed = request.changed_paths or ()
    predicted = request.predicted_files
    required_tests = request.required_test_ids or envelope.required_test_ids
    required_proofs = request.required_proof_ids or envelope.required_proof_ids
    conditions = {
        "policy_autonomous_merge_enabled": policy.autonomous_merge_enabled is True,
        "risk_class_r2_or_lower": risk in _LOW_RISK_CLASSES,
        "reversible": envelope.reversible is True and envelope.risk_assessment.reversible is True,
        "autonomy_level_admits_execution": (
            envelope.autonomy_level in _MUTATING_LEVELS
            and policy.allows(envelope.autonomy_level, risk)
        ),
        "scope_within_envelope": all(
            path_under_any(path, envelope.allowed_paths) for path in predicted
        ),
        "no_scope_escape": "scope_escape" not in forbidden_reasons
        and all(path_under_any(path, envelope.allowed_paths) for path in (*predicted, *changed)),
        "no_self_edit": "self_edit" not in forbidden_reasons
        and not any(is_self_edit_path(path) for path in (*predicted, *changed)),
        "no_validator_policy_key_mutation": "validator_policy_key" not in forbidden_reasons
        and not any(is_validator_policy_key_path(path) for path in (*predicted, *changed)),
        "exact_predicted_files": bool(predicted),
        "exact_predicted_symbols": bool(request.predicted_symbols),
        "predicted_files_within_patch_bounds": len(predicted) <= request.max_changed_files
        and request.changed_line_count <= request.max_changed_lines,
        "required_tests_satisfied": set(required_tests).issubset(request.completed_test_ids),
        "required_proofs_satisfied": set(required_proofs).issubset(request.completed_proof_ids),
        "validation_receipts_current": bool(request.validation_receipt_ids)
        and (not required_tests or bool(request.validation_receipt_ids)),
        "isolated_worktree_bound": bool(request.worktree_id),
        "rollback_plan_bound": bool(request.rollback_plan_id),
        "terminal_succeeded": request.terminal_status is TerminalStatus.SUCCEEDED
        and disposition is RepairControllerDisposition.EXECUTED,
        "repair_receipt_does_not_authorize_merge": receipt is None or receipt.authorizes_merge is False,
        "not_identical_failure_backoff": disposition
        not in {
            RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF,
            RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED,
        },
        "not_r3_or_higher": risk.rank < RiskClass.R3_BOUNDED_REPOSITORY_MUTATION.rank,
        "changed_paths_within_envelope": bool(changed)
        and all(path_under_any(path, envelope.allowed_paths) for path in changed),
        "changed_paths_subset_of_predicted": bool(changed) and set(changed).issubset(predicted),
    }
    return {name: bool(conditions[name]) for name in LOW_RISK_MERGE_CONDITIONS}


def merge_disposition_for(
    *,
    conditions: Mapping[str, bool],
    risk: RiskClass,
    forbidden_reasons: Sequence[str],
    terminal_status: TerminalStatus,
) -> RepairMergeDisposition:
    if forbidden_reasons:
        return RepairMergeDisposition.REJECTED
    if terminal_status is TerminalStatus.FAILED:
        return RepairMergeDisposition.ROLLBACK
    if risk.rank >= RiskClass.R3_BOUNDED_REPOSITORY_MUTATION.rank:
        return RepairMergeDisposition.PROPOSAL_REQUIRED
    if all(conditions.get(name, False) for name in LOW_RISK_MERGE_CONDITIONS):
        return RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    return RepairMergeDisposition.HOLD


class AutonomousRepairController:
    """Tier-selection and scope facade over ``AutonomousRepairEngine``.

    The controller never constructs a substitute engine type, never writes
    source bytes itself, and never authorizes merge.  ``DecisionRuntime`` and
    existing merge/control services remain the effect boundary.
    """

    def __init__(
        self,
        *,
        envelope: AutonomyEnvelope,
        policy: AutonomyPolicy,
        engine: AutonomousRepairEngine | None = None,
        repo_root: str | Path | None = None,
        model_invoker: ModelInvoker | None = None,
        backoff_policy: RepairBackoffPolicy | None = None,
    ) -> None:
        if not isinstance(envelope, AutonomyEnvelope):
            raise RepairControllerError("envelope must be an AutonomyEnvelope")
        if not isinstance(policy, AutonomyPolicy):
            raise RepairControllerError("policy must be an AutonomyPolicy")
        if envelope.policy_id != policy.policy_id:
            raise RepairControllerError("envelope policy identity does not match the bound policy")
        if engine is not None and type(engine) is not CANONICAL_REPAIR_ENGINE_TYPE:
            raise RepairControllerError(
                "repair facade must compose the existing AutonomousRepairEngine"
            )
        if engine is None and repo_root is not None:
            engine = AutonomousRepairEngine(repo_root=repo_root)
            if type(engine) is not CANONICAL_REPAIR_ENGINE_TYPE:
                raise RepairControllerError("repair facade cannot substitute a second repair engine")
        self._envelope = envelope
        self._policy = policy
        self._engine = engine
        self._repo_root = None if repo_root is None else Path(repo_root).resolve()
        self._model_invoker = model_invoker
        self._backoff = backoff_policy or RepairBackoffPolicy()
        self._failures: dict[str, RepairFailureRecord] = {}
        self._attempt_count = 0
        self._model_call_count = 0

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine_interface(self) -> str:
        return CANONICAL_REPAIR_ENGINE_INTERFACE

    @property
    def envelope(self) -> AutonomyEnvelope:
        return self._envelope

    @property
    def policy(self) -> AutonomyPolicy:
        return self._policy

    @property
    def engine(self) -> AutonomousRepairEngine | None:
        return self._engine

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    @property
    def authorizes_effect(self) -> bool:
        return False

    @property
    def authorizes_merge(self) -> bool:
        return False

    def _result(
        self,
        *,
        disposition: RepairControllerDisposition,
        request: RepairRequest,
        selected_tier: RepairTier | None = None,
        plan: AutonomousRepairPlan | None = None,
        receipt: AutonomousRepairReceipt | None = None,
        reason_codes: Sequence[str] = (),
        model_call_count: int = 0,
        diagnostic_reused: bool = False,
        backoff_milliseconds: int = 0,
        forbidden_reasons: Sequence[str] = (),
        engine_report_passed: bool = False,
        source_edit_admitted: bool = False,
        diagnostic_receipt_id: str = "",
    ) -> RepairControllerResult:
        conditions = evaluate_merge_conditions(
            policy=self._policy,
            envelope=self._envelope,
            request=request,
            disposition=disposition,
            receipt=receipt,
            forbidden_reasons=forbidden_reasons,
        )
        merge = merge_disposition_for(
            conditions=conditions,
            risk=self._envelope.risk_assessment.risk_class,
            forbidden_reasons=forbidden_reasons,
            terminal_status=request.terminal_status,
        )
        if disposition in {
            RepairControllerDisposition.REJECTED_SCOPE_ESCAPE,
            RepairControllerDisposition.REJECTED_SELF_EDIT,
            RepairControllerDisposition.REJECTED_VALIDATOR_POLICY_KEY,
            RepairControllerDisposition.REJECTED_MISSING_PRECONDITIONS,
            RepairControllerDisposition.REJECTED_UNADMITTED_SOURCE_EDIT,
            RepairControllerDisposition.REJECTED_SECOND_ENGINE,
        }:
            merge = RepairMergeDisposition.REJECTED
        elif disposition is RepairControllerDisposition.ROLLBACK_REQUIRED:
            merge = RepairMergeDisposition.ROLLBACK
        elif disposition in {
            RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF,
            RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED,
            RepairControllerDisposition.REPAIR_BOUND_EXCEEDED,
        }:
            merge = RepairMergeDisposition.HOLD
        return RepairControllerResult(
            disposition=disposition,
            merge_disposition=merge,
            selected_tier=selected_tier,
            plan=plan,
            receipt=receipt,
            reason_codes=tuple(reason_codes),
            model_call_count=model_call_count,
            diagnostic_reused=diagnostic_reused,
            backoff_milliseconds=backoff_milliseconds,
            merge_condition_results=conditions,
            engine_report_passed=engine_report_passed,
            source_edit_admitted=source_edit_admitted,
            failure_signature=request.failure_signature,
            diagnostic_receipt_id=diagnostic_receipt_id or request.diagnostic_receipt_id,
        )

    def select_tier(self, request: RepairRequest | Mapping[str, Any]) -> RepairTier:
        bound = RepairRequest.from_dict(request)
        if bound.requested_tier is not None:
            return bound.requested_tier
        if bound.requires_model:
            return RepairTier.MODEL_ASSISTED_BOUNDED
        if bound.template_id:
            return RepairTier.TEMPLATE_CONSTRAINED
        return RepairTier.DETERMINISTIC

    def _precondition_reasons(self, request: RepairRequest, tier: RepairTier) -> tuple[str, ...]:
        reasons: list[str] = []
        if not request.predicted_files:
            reasons.append("exact_files_required")
        if not request.predicted_symbols:
            reasons.append("exact_symbols_required")
        if not request.context_reference_ids:
            reasons.append("sufficient_context_required")
        if not request.worktree_id:
            reasons.append("isolated_worktree_required")
        if not request.rollback_plan_id:
            reasons.append("rollback_plan_required")
        required_tests = request.required_test_ids or self._envelope.required_test_ids
        required_proofs = request.required_proof_ids or self._envelope.required_proof_ids
        if not required_tests and not required_proofs:
            reasons.append("predetermined_checks_required")
        if not self._envelope.allowed_paths:
            reasons.append("envelope_allowed_paths_required")
        if tier is RepairTier.MODEL_ASSISTED_BOUNDED and not request.worktree_id:
            reasons.append("model_assisted_requires_isolated_worktree")
        if tier is RepairTier.TEMPLATE_CONSTRAINED and not request.template_id:
            reasons.append("template_id_required")
        if not self._policy.allows(
            self._envelope.autonomy_level, self._envelope.risk_assessment.risk_class
        ):
            reasons.append("policy_forbids_requested_autonomy_level")
        return tuple(reasons)

    def _build_plan(self, request: RepairRequest, tier: RepairTier) -> AutonomousRepairPlan:
        required_tests = request.required_test_ids or self._envelope.required_test_ids
        required_proofs = request.required_proof_ids or self._envelope.required_proof_ids
        return AutonomousRepairPlan(
            objective_id=self._envelope.objective_id,
            task_id=self._envelope.task_id,
            repair_tier=tier,
            predicted_files=request.predicted_files,
            predicted_symbols=request.predicted_symbols,
            patch_envelope_id=self._envelope.envelope_id,
            context_reference_ids=request.context_reference_ids,
            required_test_ids=required_tests,
            required_proof_ids=required_proofs,
            worktree_id=request.worktree_id,
            allowed_paths=self._envelope.allowed_paths,
            forbidden_symbols=request.forbidden_symbols,
            rollback_plan_id=request.rollback_plan_id,
            risk_class=self._envelope.risk_assessment.risk_class,
            max_changed_files=request.max_changed_files,
            max_changed_lines=request.max_changed_lines,
        )

    def _build_receipt(
        self,
        *,
        plan: AutonomousRepairPlan,
        request: RepairRequest,
        terminal_status: TerminalStatus,
        failure_signature: str = "",
        diagnostic_receipt_id: str = "",
    ) -> AutonomousRepairReceipt:
        return AutonomousRepairReceipt(
            plan_id=plan.plan_id,
            envelope_id=self._envelope.envelope_id,
            terminal_status=terminal_status,
            changed_paths=request.changed_paths,
            validation_receipt_ids=request.validation_receipt_ids,
            proof_receipt_ids=request.proof_receipt_ids,
            adversarial_assurance_receipt_ids=request.adversarial_assurance_receipt_ids,
            rollback_receipt_id=request.rollback_plan_id if terminal_status is TerminalStatus.FAILED else "",
            failure_signature=failure_signature,
            diagnostic_receipt_id=diagnostic_receipt_id,
            authorizes_merge=False,
        )

    def admit_source_edit(
        self,
        operator: Mapping[str, Any] | AdmittedSourceEditOperator | None,
        *,
        preferred_path: str,
        repo_root: str | Path | None = None,
        allow_code_edit_materialize: bool = False,
    ) -> SourceEditAdmission:
        """Admit one existing exact source-edit operator.  A policy flag is not admission."""

        del allow_code_edit_materialize  # a policy flag cannot become admission
        if operator is None:
            return SourceEditAdmission(
                admitted=False,
                disposition="not_source_edit",
                reasons=("typed_admitted_source_edit_operator_required", "policy_flag_is_not_source_edit_admission"),
            )
        try:
            bound = (
                operator
                if isinstance(operator, AdmittedSourceEditOperator)
                else AdmittedSourceEditOperator.from_mapping(operator)
            )
        except AdmittedSourceEditError as exc:
            return SourceEditAdmission(
                admitted=False,
                disposition="rejected",
                reasons=(str(exc) or "source_edit_operator_not_admitted",),
            )
        if type(bound) is not CANONICAL_SOURCE_EDIT_OPERATOR_TYPE:
            return SourceEditAdmission(
                admitted=False,
                disposition="rejected",
                reasons=("second_source_edit_operator_rejected",),
            )
        path = bound.relative_path
        forbidden = classify_forbidden_paths((path,), self._envelope.allowed_paths)
        if forbidden:
            return SourceEditAdmission(
                admitted=False,
                disposition="rejected",
                reasons=forbidden,
                operator_id=bound.operator_id,
                relative_path=path,
                old_digest=bound.old_digest,
                new_digest=bound.new_digest,
            )
        if preferred_path and path != preferred_path:
            return SourceEditAdmission(
                admitted=False,
                disposition="rejected",
                reasons=("source_edit_path_binding_mismatch",),
                operator_id=bound.operator_id,
                relative_path=path,
            )
        root = repo_root if repo_root is not None else self._repo_root
        if root is not None:
            try:
                bound.validate(
                    repo_root=Path(root).resolve(),
                    preferred_path=preferred_path or path,
                )
            except AdmittedSourceEditError as exc:
                return SourceEditAdmission(
                    admitted=False,
                    disposition="rejected",
                    reasons=(str(exc) or "source_edit_operator_not_admitted",),
                    operator_id=bound.operator_id,
                    relative_path=path,
                    old_digest=bound.old_digest,
                    new_digest=bound.new_digest,
                )
        return SourceEditAdmission(
            admitted=True,
            disposition="validation_pending",
            reasons=("admitted_source_edit_validation_pending",),
            operator_id=bound.operator_id,
            relative_path=path,
            old_digest=bound.old_digest,
            new_digest=bound.new_digest,
        )

    def plan(self, request: RepairRequest | Mapping[str, Any]) -> RepairControllerResult:
        """Bind a tier and envelope without executing a model or merge."""

        bound = RepairRequest.from_dict(request)
        forbidden = classify_forbidden_paths(
            (*bound.predicted_files, *bound.changed_paths),
            self._envelope.allowed_paths,
        )
        forbidden_symbols = set(bound.forbidden_symbols).intersection(bound.predicted_symbols)
        if forbidden_symbols:
            forbidden = tuple(dict.fromkeys((*forbidden, "validator_policy_key")))
        if "self_edit" in forbidden:
            return self._result(
                disposition=RepairControllerDisposition.REJECTED_SELF_EDIT,
                request=bound,
                reason_codes=("self_edit", *forbidden),
                forbidden_reasons=forbidden,
            )
        if "validator_policy_key" in forbidden:
            return self._result(
                disposition=RepairControllerDisposition.REJECTED_VALIDATOR_POLICY_KEY,
                request=bound,
                reason_codes=("validator_policy_key", *forbidden),
                forbidden_reasons=forbidden,
            )
        if "scope_escape" in forbidden:
            return self._result(
                disposition=RepairControllerDisposition.REJECTED_SCOPE_ESCAPE,
                request=bound,
                reason_codes=("scope_escape", *forbidden),
                forbidden_reasons=forbidden,
            )
        tier = self.select_tier(bound)
        missing = self._precondition_reasons(bound, tier)
        if missing:
            return self._result(
                disposition=RepairControllerDisposition.REJECTED_MISSING_PRECONDITIONS,
                request=bound,
                selected_tier=tier,
                reason_codes=missing,
            )
        if bound.source_edit_operator is not None or bound.allow_code_edit_materialize:
            operator_path = ""
            if bound.source_edit_operator is not None:
                operator_path = str(bound.source_edit_operator.get("relative_path") or "")
            admission = self.admit_source_edit(
                bound.source_edit_operator,
                preferred_path=operator_path or bound.predicted_files[0],
                repo_root=self._repo_root,
                allow_code_edit_materialize=bound.allow_code_edit_materialize,
            )
            if not admission.admitted:
                return self._result(
                    disposition=RepairControllerDisposition.REJECTED_UNADMITTED_SOURCE_EDIT,
                    request=bound,
                    selected_tier=tier,
                    reason_codes=admission.reasons,
                    forbidden_reasons=classify_forbidden_paths(
                        (admission.relative_path,) if admission.relative_path else (),
                        self._envelope.allowed_paths,
                    ),
                )
        plan = self._build_plan(bound, tier)
        return self._result(
            disposition=RepairControllerDisposition.ADMITTED,
            request=bound,
            selected_tier=tier,
            plan=plan,
            reason_codes=("tier_selected", tier.value),
            source_edit_admitted=bound.source_edit_operator is not None,
        )

    def _remember_failure(
        self,
        request: RepairRequest,
        *,
        model_calls: int,
        diagnostic_receipt_id: str,
    ) -> RepairFailureRecord:
        signature = request.failure_signature
        previous = self._failures.get(signature)
        count = 1 if previous is None else previous.count + 1
        record = RepairFailureRecord(
            failure_signature=signature,
            diagnostic_receipt_id=diagnostic_receipt_id or request.diagnostic_receipt_id,
            count=count,
            model_call_count=(0 if previous is None else previous.model_call_count) + model_calls,
            last_backoff_milliseconds=_backoff_milliseconds(
                count,
                base=self._backoff.base_backoff_milliseconds,
                maximum=self._backoff.max_backoff_milliseconds,
            ),
        )
        self._failures[signature] = record
        return record

    def _invoke_model(self, request: RepairRequest, tier: RepairTier) -> tuple[int, str]:
        if tier is not RepairTier.MODEL_ASSISTED_BOUNDED or self._model_invoker is None:
            return 0, request.diagnostic_receipt_id
        diagnostic = self._model_invoker(request)
        if not isinstance(diagnostic, Mapping):
            raise RepairControllerError("model invoker must return an object")
        _reject_forbidden_keys(diagnostic, "model diagnostic")
        self._model_call_count += 1
        receipt_id = _identifier(
            diagnostic.get("diagnostic_receipt_id") or request.diagnostic_receipt_id,
            "diagnostic_receipt_id",
            required=False,
        )
        return 1, receipt_id

    def execute(self, request: RepairRequest | Mapping[str, Any]) -> RepairControllerResult:
        """Select a tier, optionally delegate to the existing engine, and emit evidence."""

        bound = RepairRequest.from_dict(request)
        planned = self.plan(bound)
        if planned.disposition is not RepairControllerDisposition.ADMITTED:
            return planned
        assert planned.plan is not None
        assert planned.selected_tier is not None
        self._attempt_count += 1
        max_rounds = self._envelope.cognitive_budget.max_repair_rounds
        if self._attempt_count > max_rounds:
            return self._result(
                disposition=RepairControllerDisposition.REPAIR_BOUND_EXCEEDED,
                request=bound,
                selected_tier=planned.selected_tier,
                plan=planned.plan,
                reason_codes=("max_repair_rounds_exceeded",),
            )

        signature = bound.failure_signature
        previous = self._failures.get(signature) if signature else None
        if previous is not None:
            record = self._remember_failure(
                bound, model_calls=0, diagnostic_receipt_id=previous.diagnostic_receipt_id
            )
            exhausted = record.count > self._backoff.max_identical_failures
            receipt = self._build_receipt(
                plan=planned.plan,
                request=bound,
                terminal_status=TerminalStatus.BLOCKED,
                failure_signature=signature,
                diagnostic_receipt_id=record.diagnostic_receipt_id,
            )
            return self._result(
                disposition=(
                    RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
                    if exhausted
                    else RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
                ),
                request=bound,
                selected_tier=planned.selected_tier,
                plan=planned.plan,
                receipt=receipt,
                reason_codes=(
                    "identical_failure",
                    "diagnostic_reused",
                    "model_call_suppressed",
                ),
                model_call_count=0,
                diagnostic_reused=True,
                backoff_milliseconds=record.last_backoff_milliseconds,
                diagnostic_receipt_id=record.diagnostic_receipt_id,
            )

        if self._engine is not None and type(self._engine) is not CANONICAL_REPAIR_ENGINE_TYPE:
            return self._result(
                disposition=RepairControllerDisposition.REJECTED_SECOND_ENGINE,
                request=bound,
                selected_tier=planned.selected_tier,
                reason_codes=("second_repair_engine_rejected",),
            )

        model_calls, diagnostic_id = self._invoke_model(bound, planned.selected_tier)
        engine_passed = False
        if bound.work_items:
            if self._engine is None:
                raise RepairControllerError(
                    "existing AutonomousRepairEngine is required to execute work items"
                )
            report = self._engine.run(list(bound.work_items))
            engine_passed = bool(getattr(report, "passed", False))
            model_calls += int(getattr(report, "model_call_count", 0) or 0)
            self._model_call_count += int(getattr(report, "model_call_count", 0) or 0)

        terminal = bound.terminal_status
        if terminal is TerminalStatus.FAILED:
            if signature:
                self._remember_failure(
                    bound, model_calls=model_calls, diagnostic_receipt_id=diagnostic_id
                )
            receipt = self._build_receipt(
                plan=planned.plan,
                request=bound,
                terminal_status=TerminalStatus.FAILED,
                failure_signature=signature,
                diagnostic_receipt_id=diagnostic_id,
            )
            return self._result(
                disposition=RepairControllerDisposition.ROLLBACK_REQUIRED,
                request=bound,
                selected_tier=planned.selected_tier,
                plan=planned.plan,
                receipt=receipt,
                reason_codes=("rollback_plan_bound", bound.rollback_plan_id),
                model_call_count=model_calls,
                diagnostic_receipt_id=diagnostic_id,
                engine_report_passed=engine_passed,
                source_edit_admitted=planned.source_edit_admitted,
            )

        receipt_status = (
            TerminalStatus.SUCCEEDED
            if terminal is TerminalStatus.SUCCEEDED
            else TerminalStatus.PENDING
        )
        if receipt_status is TerminalStatus.SUCCEEDED and not bound.validation_receipt_ids:
            receipt_status = TerminalStatus.PENDING
        execute_request = bound
        if receipt_status is TerminalStatus.PENDING:
            execute_request = RepairRequest.from_dict(
                {**bound.to_dict(), "terminal_status": TerminalStatus.PENDING.value}
            )
        receipt = self._build_receipt(
            plan=planned.plan,
            request=execute_request,
            terminal_status=receipt_status,
            failure_signature=signature,
            diagnostic_receipt_id=diagnostic_id,
        )
        return self._result(
            disposition=RepairControllerDisposition.EXECUTED,
            request=execute_request,
            selected_tier=planned.selected_tier,
            plan=planned.plan,
            receipt=receipt,
            reason_codes=("executed_through_existing_engine", planned.selected_tier.value),
            model_call_count=model_calls,
            diagnostic_receipt_id=diagnostic_id,
            engine_report_passed=engine_passed,
            source_edit_admitted=planned.source_edit_admitted,
        )

    def snapshot(self) -> Mapping[str, Any]:
        payload = {
            "schema": REPAIR_CONTROLLER_SNAPSHOT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "envelope": self._envelope.to_dict(),
            "policy": self._policy.to_dict(),
            "attempt_count": self._attempt_count,
            "model_call_count": self._model_call_count,
            "failures": [record.to_dict() for record in self._failures.values()],
            "backoff_policy": {
                "base_backoff_milliseconds": self._backoff.base_backoff_milliseconds,
                "max_backoff_milliseconds": self._backoff.max_backoff_milliseconds,
                "max_identical_failures": self._backoff.max_identical_failures,
            },
            "engine_bound": self._engine is not None,
            "engine_interface": CANONICAL_REPAIR_ENGINE_INTERFACE,
            "authorizes_effect": False,
            "authorizes_merge": False,
        }
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > MAX_REPAIR_CONTROLLER_SNAPSHOT_BYTES:
            raise RepairControllerError("repair controller snapshot exceeds its bounded size")
        payload["snapshot_id"] = content_identity({k: v for k, v in payload.items() if k != "snapshot_id"})
        return MappingProxyType(payload)

    def snapshot_json(self) -> str:
        return canonical_json(dict(self.snapshot()))


__all__ = [
    "AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE",
    "AUTONOMOUS_REPAIR_CONTROLLER_SCHEMA",
    "CANONICAL_REPAIR_ENGINE_INTERFACE",
    "CANONICAL_REPAIR_ENGINE_TYPE",
    "CANONICAL_SOURCE_EDIT_OPERATOR_TYPE",
    "CONTROLLER_RELATIVE_PATH",
    "LOW_RISK_MERGE_CONDITIONS",
    "PROTECTED_AUTHORITY_PATHS",
    "REPAIR_CONTROLLER_RESULT_SCHEMA",
    "SELF_PROTECTING_PATHS",
    "SOURCE_EDIT_ADMISSION_SCHEMA",
    "AutonomousRepairController",
    "RepairBackoffPolicy",
    "RepairControllerDisposition",
    "RepairControllerError",
    "RepairControllerResult",
    "RepairFailureRecord",
    "RepairMergeDisposition",
    "RepairRequest",
    "SourceEditAdmission",
    "classify_forbidden_paths",
    "evaluate_merge_conditions",
    "is_self_edit_path",
    "is_validator_policy_key_path",
    "merge_disposition_for",
    "path_under_any",
    "path_under_prefix",
]
