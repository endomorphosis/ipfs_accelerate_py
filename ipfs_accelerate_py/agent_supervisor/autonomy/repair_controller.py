# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded facade over the existing autonomous-repair engine.

``AutonomousRepairController@1`` selects among deterministic, template-
constrained, and model-assisted tiers and binds one admitted envelope,
isolated worktree, predetermined checks, backoff, and merge disposition.
It is not a second repair engine, materializer, DecisionRuntime, worktree
manager, or merge authority.

Admission and execution stay on
``agent_supervisor.autonomous_repair.engine.AutonomousRepairEngine`` and
``AdmittedSourceEditOperator``.  Repair receipts remain evidence: they never
independently authorize merge or an effect.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
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
    "ipfs_accelerate_py/agent-supervisor/autonomy/autonomous-repair-controller@1"
)
REPAIR_CONTROLLER_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-request@1"
)
REPAIR_CONTROLLER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-result@1"
)
REPAIR_MERGE_CONJUNCTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-merge-conjunction@1"
)

SELF_EDIT_PATHS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair/",
)

VALIDATOR_POLICY_KEY_PATHS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py/agent_supervisor/validation/",
    "ipfs_accelerate_py/agent_supervisor/proof/",
    "ipfs_accelerate_py/agent_supervisor/verification/",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py",
    "ipfs_accelerate_py/llm_router.py",
    "config/",
    "secrets/",
    "credentials/",
    ".ssh/",
    ".aws/",
    ".gnupg/",
    ".env",
)

_VALIDATOR_POLICY_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "authority_policy",
        "policy_key",
        "promotion_rule",
        "signing_key",
        "trusted_key",
        "validator_policy",
        "validation_policy",
    }
)

PROTECTED_AUTHORITY_PATHS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*SELF_EDIT_PATHS, *VALIDATOR_POLICY_KEY_PATHS))
)

LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "autonomous_merge_enabled",
    "risk_at_most_r2",
    "reversible",
    "autonomy_level_permits",
    "policy_allows_execute_reversible",
    "exact_paths_bound",
    "exact_symbols_bound",
    "isolated_worktree",
    "predetermined_tests",
    "predetermined_proofs",
    "rollback_plan",
    "protected_authority_paths_intact",
    "not_self_edit",
    "not_scope_escape",
    "not_validator_policy_key_mutation",
    "current_validation_evidence",
    "repair_succeeded",
    "receipt_does_not_authorize_merge",
)

DEFAULT_BASE_BACKOFF_MILLISECONDS: Final[int] = 10
DEFAULT_MAX_BACKOFF_MILLISECONDS: Final[int] = 40
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 4
DEFAULT_MAX_CHANGED_FILES: Final[int] = 8
DEFAULT_MAX_CHANGED_LINES: Final[int] = 200

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

_LOW_RISK: Final[frozenset[RiskClass]] = frozenset(
    {
        RiskClass.R0_PURE,
        RiskClass.R1_READ_ONLY,
        RiskClass.R2_REVERSIBLE_LOCAL,
    }
)


class RepairControllerError(ValueError):
    """Raised when repair-controller inputs violate the frozen facade contract."""


class RepairControllerDisposition(str, Enum):
    """Closed outcome of one facade evaluation."""

    ADMITTED = "admitted"
    ENGINE_DELEGATED = "engine_delegated"
    REJECTED = "rejected"
    IDENTICAL_FAILURE_REUSED = "identical_failure_reused"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class RepairMergeDisposition(str, Enum):
    """Closed merge recommendation; never itself a merge grant."""

    HOLD = "hold"
    REJECTED = "rejected"
    PROPOSAL = "proposal"
    AUTONOMOUS_MERGE = "autonomous_merge"


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


def _paths(
    value: Any,
    name: str,
    *,
    required: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    identifiers = _identifiers(value, name, required=required, preserve_order=True)
    normalized = tuple(_posix_path(item, name) for item in identifiers)
    if preserve_order:
        return normalized
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


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int = MAX_IDENTIFIER_BYTES * 8) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise RepairControllerError(f"{name} must be a bounded integer")
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


def _path_under(path: str, prefix: str) -> bool:
    candidate = path.replace("\\", "/").rstrip("/")
    bound = prefix.replace("\\", "/").rstrip("/")
    if not candidate or not bound:
        return False
    return candidate == bound or candidate.startswith(bound + "/")


def path_escapes_allowed(path: str, allowed_paths: Sequence[str]) -> bool:
    """Return True when ``path`` is outside every admitted prefix."""

    return not any(_path_under(path, prefix) for prefix in allowed_paths)


def path_is_self_edit(path: str) -> bool:
    """Return True when ``path`` would edit the repair facade or engine."""

    return any(_path_under(path, prefix) for prefix in SELF_EDIT_PATHS)


def path_is_validator_policy_key(path: str) -> bool:
    """Return True when ``path`` mutates validator, policy, or key authority."""

    if any(_path_under(path, prefix) for prefix in VALIDATOR_POLICY_KEY_PATHS):
        return True
    stem = PurePosixPath(path).name.lower().replace("-", "_")
    collapsed = path.lower().replace("-", "_")
    return any(marker in stem or marker in collapsed for marker in _VALIDATOR_POLICY_KEY_MARKERS)


def classify_repair_path(path: str, allowed_paths: Sequence[str]) -> tuple[str, ...]:
    """Return closed rejection codes for one predicted or changed path."""

    codes: list[str] = []
    if path_escapes_allowed(path, allowed_paths):
        codes.append("scope_escape")
    if path_is_self_edit(path):
        codes.append("self_edit")
    if path_is_validator_policy_key(path):
        codes.append("validator_policy_key_mutation")
    return tuple(codes)


def _backoff_milliseconds(count: int, *, base: int, maximum: int) -> int:
    if count <= 0:
        return 0
    shift = min(count - 1, 16)
    value = base * (2**shift)
    return maximum if value > maximum else value


@dataclass(frozen=True)
class RepairFailureRecord:
    """One bounded identical-failure diagnosis retained by the facade."""

    signature: str
    count: int
    diagnostic_receipt_id: str
    backoff_milliseconds: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "signature", _identifier(self.signature, "signature"))
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _identifier(self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False),
        )
        object.__setattr__(self, "count", _int(self.count, "count", minimum=1, maximum=1_000_000))
        object.__setattr__(
            self,
            "backoff_milliseconds",
            _int(self.backoff_milliseconds, "backoff_milliseconds", maximum=86_400_000),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "count": self.count,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "backoff_milliseconds": self.backoff_milliseconds,
        }


@dataclass(frozen=True)
class RepairControllerRequest:
    """One envelope-bound repair attempt presented to the facade."""

    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    worktree_id: str
    rollback_plan_id: str
    requested_tier: RepairTier | None = None
    context_reference_ids: tuple[str, ...] = ()
    required_test_ids: tuple[str, ...] = ()
    required_proof_ids: tuple[str, ...] = ()
    forbidden_symbols: tuple[str, ...] = ()
    max_changed_files: int = DEFAULT_MAX_CHANGED_FILES
    max_changed_lines: int = DEFAULT_MAX_CHANGED_LINES
    failure_signature: str = ""
    new_evidence_since_failure: bool = False
    source_edit_operator: Mapping[str, Any] | None = None
    claimed_applied: bool = False
    changed_paths: tuple[str, ...] = ()
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    diagnostic_receipt_id: str = ""
    operation: str = "bounded_repair"
    work_id: str = ""
    template_available: bool = False
    require_model: bool = False
    suffix_receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_files",
            _paths(self.predicted_files, "predicted_files", required=True, preserve_order=True),
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _identifiers(self.predicted_symbols, "predicted_symbols", required=True),
        )
        object.__setattr__(self, "worktree_id", _identifier(self.worktree_id, "worktree_id"))
        object.__setattr__(
            self, "rollback_plan_id", _identifier(self.rollback_plan_id, "rollback_plan_id")
        )
        if self.requested_tier is not None:
            object.__setattr__(
                self,
                "requested_tier",
                _enum(self.requested_tier, RepairTier, "requested_tier"),
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
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
        object.__setattr__(
            self, "changed_paths", _paths(self.changed_paths, "changed_paths", preserve_order=True)
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
            self,
            "failure_signature",
            _identifier(self.failure_signature, "failure_signature", required=False),
        )
        object.__setattr__(
            self,
            "new_evidence_since_failure",
            _bool(self.new_evidence_since_failure, "new_evidence_since_failure"),
        )
        object.__setattr__(self, "claimed_applied", _bool(self.claimed_applied, "claimed_applied"))
        object.__setattr__(
            self,
            "diagnostic_receipt_id",
            _identifier(self.diagnostic_receipt_id, "diagnostic_receipt_id", required=False),
        )
        object.__setattr__(self, "operation", _identifier(self.operation, "operation"))
        object.__setattr__(self, "work_id", _identifier(self.work_id, "work_id", required=False))
        object.__setattr__(
            self, "template_available", _bool(self.template_available, "template_available")
        )
        object.__setattr__(self, "require_model", _bool(self.require_model, "require_model"))
        object.__setattr__(
            self,
            "suffix_receipt_id",
            _identifier(self.suffix_receipt_id, "suffix_receipt_id", required=False),
        )
        if self.source_edit_operator is not None:
            if not isinstance(self.source_edit_operator, Mapping):
                raise RepairControllerError("source_edit_operator must be an object")
            if len(self.source_edit_operator) > MAX_MAPPING_ITEMS:
                raise RepairControllerError("source_edit_operator contains too many fields")
            _reject_forbidden_keys(self.source_edit_operator, "source_edit_operator")
        if len(self.predicted_files) > self.max_changed_files:
            raise RepairControllerError("predicted repair files exceed the patch envelope")

    @property
    def scoped_paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.predicted_files, *self.changed_paths)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_CONTROLLER_REQUEST_SCHEMA,
            "predicted_files": list(self.predicted_files),
            "predicted_symbols": list(self.predicted_symbols),
            "worktree_id": self.worktree_id,
            "rollback_plan_id": self.rollback_plan_id,
            "requested_tier": None if self.requested_tier is None else self.requested_tier.value,
            "context_reference_ids": list(self.context_reference_ids),
            "required_test_ids": list(self.required_test_ids),
            "required_proof_ids": list(self.required_proof_ids),
            "forbidden_symbols": list(self.forbidden_symbols),
            "max_changed_files": self.max_changed_files,
            "max_changed_lines": self.max_changed_lines,
            "failure_signature": self.failure_signature,
            "new_evidence_since_failure": self.new_evidence_since_failure,
            "source_edit_operator": (
                None if self.source_edit_operator is None else dict(self.source_edit_operator)
            ),
            "claimed_applied": self.claimed_applied,
            "changed_paths": list(self.changed_paths),
            "validation_receipt_ids": list(self.validation_receipt_ids),
            "proof_receipt_ids": list(self.proof_receipt_ids),
            "adversarial_assurance_receipt_ids": list(self.adversarial_assurance_receipt_ids),
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "operation": self.operation,
            "work_id": self.work_id,
            "template_available": self.template_available,
            "require_model": self.require_model,
            "suffix_receipt_id": self.suffix_receipt_id,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | RepairControllerRequest) -> RepairControllerRequest:
        if isinstance(payload, RepairControllerRequest):
            return payload
        if not isinstance(payload, Mapping):
            raise RepairControllerError("repair request must be an object")
        _reject_forbidden_keys(payload, "repair request")
        expected = {
            "schema",
            "predicted_files",
            "predicted_symbols",
            "worktree_id",
            "rollback_plan_id",
            "requested_tier",
            "context_reference_ids",
            "required_test_ids",
            "required_proof_ids",
            "forbidden_symbols",
            "max_changed_files",
            "max_changed_lines",
            "failure_signature",
            "new_evidence_since_failure",
            "source_edit_operator",
            "claimed_applied",
            "changed_paths",
            "validation_receipt_ids",
            "proof_receipt_ids",
            "adversarial_assurance_receipt_ids",
            "diagnostic_receipt_id",
            "operation",
            "work_id",
            "template_available",
            "require_model",
            "suffix_receipt_id",
        }
        extra = set(payload).difference(expected)
        if extra:
            raise RepairControllerError("repair request contains unsupported fields")
        if payload.get("schema") not in (None, "", REPAIR_CONTROLLER_REQUEST_SCHEMA):
            raise RepairControllerError("unsupported repair request schema")
        return cls(
            predicted_files=tuple(payload.get("predicted_files") or ()),
            predicted_symbols=tuple(payload.get("predicted_symbols") or ()),
            worktree_id=payload.get("worktree_id", ""),
            rollback_plan_id=payload.get("rollback_plan_id", ""),
            requested_tier=payload.get("requested_tier"),
            context_reference_ids=tuple(payload.get("context_reference_ids") or ()),
            required_test_ids=tuple(payload.get("required_test_ids") or ()),
            required_proof_ids=tuple(payload.get("required_proof_ids") or ()),
            forbidden_symbols=tuple(payload.get("forbidden_symbols") or ()),
            max_changed_files=payload.get("max_changed_files", DEFAULT_MAX_CHANGED_FILES),
            max_changed_lines=payload.get("max_changed_lines", DEFAULT_MAX_CHANGED_LINES),
            failure_signature=payload.get("failure_signature", ""),
            new_evidence_since_failure=payload.get("new_evidence_since_failure", False),
            source_edit_operator=payload.get("source_edit_operator"),
            claimed_applied=payload.get("claimed_applied", False),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            validation_receipt_ids=tuple(payload.get("validation_receipt_ids") or ()),
            proof_receipt_ids=tuple(payload.get("proof_receipt_ids") or ()),
            adversarial_assurance_receipt_ids=tuple(
                payload.get("adversarial_assurance_receipt_ids") or ()
            ),
            diagnostic_receipt_id=payload.get("diagnostic_receipt_id", ""),
            operation=payload.get("operation", "bounded_repair"),
            work_id=payload.get("work_id", ""),
            template_available=payload.get("template_available", False),
            require_model=payload.get("require_model", False),
            suffix_receipt_id=payload.get("suffix_receipt_id", ""),
        )


@dataclass(frozen=True)
class RepairMergeConjunction:
    """Exact low-risk merge conjunction; every condition must hold."""

    conditions: Mapping[str, bool]
    satisfied: bool
    risk_class: RiskClass
    disposition: RepairMergeDisposition

    def __post_init__(self) -> None:
        if not isinstance(self.conditions, Mapping):
            raise RepairControllerError("merge conditions must be an object")
        if tuple(self.conditions) != LOW_RISK_MERGE_CONDITIONS:
            raise RepairControllerError("merge conjunction must use the closed condition set")
        normalized = {name: _bool(self.conditions[name], name) for name in LOW_RISK_MERGE_CONDITIONS}
        object.__setattr__(self, "conditions", MappingProxyType(normalized))
        object.__setattr__(self, "satisfied", _bool(self.satisfied, "satisfied"))
        object.__setattr__(self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairMergeDisposition, "disposition"),
        )
        expected = all(normalized.values())
        if self.satisfied != expected:
            raise RepairControllerError("merge conjunction satisfaction is inconsistent")
        if self.satisfied and self.disposition is not RepairMergeDisposition.AUTONOMOUS_MERGE:
            raise RepairControllerError("a satisfied conjunction must recommend autonomous merge")
        if (
            not self.satisfied
            and self.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE
        ):
            raise RepairControllerError("autonomous merge requires every low-risk condition")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_MERGE_CONJUNCTION_SCHEMA,
            "conditions": {name: bool(self.conditions[name]) for name in LOW_RISK_MERGE_CONDITIONS},
            "satisfied": self.satisfied,
            "risk_class": self.risk_class.value,
            "disposition": self.disposition.value,
        }


@dataclass(frozen=True)
class RepairControllerResult:
    """Closed facade outcome for one repair attempt."""

    disposition: RepairControllerDisposition
    repair_tier: RepairTier | None
    merge: RepairMergeConjunction
    reason_codes: tuple[str, ...]
    plan: AutonomousRepairPlan | None = None
    receipt: AutonomousRepairReceipt | None = None
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    model_call_count: int = 0
    engine_report: Mapping[str, Any] | None = None
    source_edit_operator_id: str = ""
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairControllerDisposition, "disposition"),
        )
        if self.repair_tier is not None:
            object.__setattr__(
                self, "repair_tier", _enum(self.repair_tier, RepairTier, "repair_tier")
            )
        if not isinstance(self.merge, RepairMergeConjunction):
            raise RepairControllerError("result merge conjunction is required")
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", required=True, preserve_order=True),
        )
        if self.plan is not None and not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan")
        if self.receipt is not None and not isinstance(self.receipt, AutonomousRepairReceipt):
            raise RepairControllerError("receipt must be an AutonomousRepairReceipt")
        object.__setattr__(
            self, "diagnostic_reused", _bool(self.diagnostic_reused, "diagnostic_reused")
        )
        object.__setattr__(
            self,
            "backoff_milliseconds",
            _int(self.backoff_milliseconds, "backoff_milliseconds", maximum=86_400_000),
        )
        object.__setattr__(
            self,
            "model_call_count",
            _int(self.model_call_count, "model_call_count", maximum=1_000_000),
        )
        if self.engine_report is not None:
            if not isinstance(self.engine_report, Mapping):
                raise RepairControllerError("engine_report must be an object")
            _reject_forbidden_keys(self.engine_report, "engine_report")
        object.__setattr__(
            self,
            "source_edit_operator_id",
            _identifier(self.source_edit_operator_id, "source_edit_operator_id", required=False),
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
        if self.receipt is not None and self.receipt.authorizes_merge:
            raise RepairControllerError("repair receipts cannot independently authorize merge")
        if (
            self.disposition is RepairControllerDisposition.REJECTED
            and self.merge.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE
        ):
            raise RepairControllerError("a rejected repair cannot be merge-eligible")
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
            "engine_interface": AUTONOMOUS_REPAIR_INTERFACE,
            "disposition": self.disposition.value,
            "repair_tier": None if self.repair_tier is None else self.repair_tier.value,
            "merge": self.merge.to_dict(),
            "reason_codes": list(self.reason_codes),
            "plan": None if self.plan is None else self.plan.to_dict(),
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "model_call_count": self.model_call_count,
            "engine_report": None if self.engine_report is None else dict(self.engine_report),
            "source_edit_operator_id": self.source_edit_operator_id,
            "failure_signature": self.failure_signature,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "authorizes_effect": False,
            "authorizes_merge": False,
        }
        if include_identity:
            payload["result_id"] = self.result_id
        return payload


def evaluate_merge_conjunction(
    *,
    policy: AutonomyPolicy,
    envelope: AutonomyEnvelope,
    request: RepairControllerRequest,
    rejection_codes: Sequence[str],
    terminal_status: TerminalStatus,
    receipt_authorizes_merge: bool,
) -> RepairMergeConjunction:
    """Evaluate every stated low-risk merge condition as a conjunction."""

    risk = envelope.risk_assessment.risk_class
    tests = request.required_test_ids or envelope.required_test_ids
    proofs = request.required_proof_ids or envelope.required_proof_ids
    scoped = request.scoped_paths
    conditions = {
        "autonomous_merge_enabled": policy.autonomous_merge_enabled,
        "risk_at_most_r2": risk in _LOW_RISK,
        "reversible": envelope.reversible and envelope.risk_assessment.reversible,
        "autonomy_level_permits": envelope.autonomy_level.rank
        >= AutonomyLevel.EXECUTE_REVERSIBLE.rank,
        "policy_allows_execute_reversible": policy.allows(
            AutonomyLevel.EXECUTE_REVERSIBLE, risk
        ),
        "exact_paths_bound": bool(request.predicted_files)
        and not any(path_escapes_allowed(path, envelope.allowed_paths) for path in scoped),
        "exact_symbols_bound": bool(request.predicted_symbols)
        and (
            not envelope.allowed_symbols
            or set(request.predicted_symbols).issubset(envelope.allowed_symbols)
        ),
        "isolated_worktree": bool(request.worktree_id),
        "predetermined_tests": bool(tests),
        "predetermined_proofs": not proofs or bool(request.proof_receipt_ids),
        "rollback_plan": bool(request.rollback_plan_id),
        "protected_authority_paths_intact": not any(
            path_is_self_edit(path) or path_is_validator_policy_key(path) for path in scoped
        ),
        "not_self_edit": "self_edit" not in rejection_codes
        and not any(path_is_self_edit(path) for path in scoped),
        "not_scope_escape": "scope_escape" not in rejection_codes
        and not any(path_escapes_allowed(path, envelope.allowed_paths) for path in scoped),
        "not_validator_policy_key_mutation": "validator_policy_key_mutation" not in rejection_codes
        and not any(path_is_validator_policy_key(path) for path in scoped),
        "current_validation_evidence": bool(request.validation_receipt_ids)
        and terminal_status is TerminalStatus.SUCCEEDED,
        "repair_succeeded": terminal_status is TerminalStatus.SUCCEEDED,
        "receipt_does_not_authorize_merge": receipt_authorizes_merge is False,
    }
    if rejection_codes:
        conditions["repair_succeeded"] = False
        conditions["current_validation_evidence"] = False
    satisfied = all(conditions[name] for name in LOW_RISK_MERGE_CONDITIONS)
    if satisfied:
        disposition = RepairMergeDisposition.AUTONOMOUS_MERGE
    elif rejection_codes:
        disposition = RepairMergeDisposition.REJECTED
    elif risk is RiskClass.R3_BOUNDED_REPOSITORY_MUTATION or risk in _LOW_RISK:
        disposition = RepairMergeDisposition.PROPOSAL
    else:
        disposition = RepairMergeDisposition.HOLD
    return RepairMergeConjunction(
        conditions=conditions,
        satisfied=satisfied,
        risk_class=risk,
        disposition=disposition,
    )


class AutonomousRepairController:
    """Tier-selection and scope facade over ``AutonomousRepairEngine``.

    Effect execution remains ``DecisionRuntime``.  Merge remains the existing
    merge-queue authority.  This object never mints a second engine.
    """

    INTERFACE: Final[str] = AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    ENGINE_INTERFACE: Final[str] = AUTONOMOUS_REPAIR_INTERFACE

    def __init__(
        self,
        *,
        envelope: AutonomyEnvelope,
        policy: AutonomyPolicy,
        engine: AutonomousRepairEngine | None = None,
        repo_root: str | None = None,
        model_call: Callable[..., Mapping[str, Any]] | None = None,
        base_backoff_milliseconds: int = DEFAULT_BASE_BACKOFF_MILLISECONDS,
        max_backoff_milliseconds: int = DEFAULT_MAX_BACKOFF_MILLISECONDS,
        max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES,
        forbidden_symbols: Sequence[str] = ("trusted_keys",),
    ) -> None:
        if not isinstance(envelope, AutonomyEnvelope):
            raise RepairControllerError("envelope must be an AutonomyEnvelope")
        if not isinstance(policy, AutonomyPolicy):
            raise RepairControllerError("policy must be an AutonomyPolicy")
        if envelope.policy_id != policy.policy_id:
            raise RepairControllerError("envelope policy_id does not match policy")
        if engine is not None:
            if type(engine) is not AutonomousRepairEngine:
                raise RepairControllerError(
                    "engine must be the existing AutonomousRepairEngine"
                )
        elif repo_root is not None:
            engine = AutonomousRepairEngine(repo_root=repo_root)
        if model_call is not None and not callable(model_call):
            raise RepairControllerError("model_call must be callable")
        self._envelope = envelope
        self._policy = policy
        self._engine = engine
        self._model_call = model_call
        self._base_backoff = _int(
            base_backoff_milliseconds, "base_backoff_milliseconds", minimum=1, maximum=86_400_000
        )
        self._max_backoff = _int(
            max_backoff_milliseconds, "max_backoff_milliseconds", minimum=1, maximum=86_400_000
        )
        self._max_identical = _int(
            max_identical_failures, "max_identical_failures", minimum=1, maximum=1_000
        )
        self._forbidden_symbols = _identifiers(forbidden_symbols, "forbidden_symbols")
        self._failures: dict[str, RepairFailureRecord] = {}
        self._model_calls = 0

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine_interface(self) -> str:
        return AUTONOMOUS_REPAIR_INTERFACE

    @property
    def engine(self) -> AutonomousRepairEngine | None:
        return self._engine

    @property
    def envelope(self) -> AutonomyEnvelope:
        return self._envelope

    @property
    def policy(self) -> AutonomyPolicy:
        return self._policy

    @property
    def model_call_count(self) -> int:
        return self._model_calls

    def select_tier(self, request: RepairControllerRequest) -> RepairTier:
        """Choose the cheapest admissible repair tier for one request."""

        bound = RepairControllerRequest.from_mapping(request)
        requested = bound.requested_tier
        if requested is RepairTier.MODEL_ASSISTED_BOUNDED or bound.require_model:
            self._require_model_assisted_preconditions(bound)
            return RepairTier.MODEL_ASSISTED_BOUNDED
        if requested is RepairTier.TEMPLATE_CONSTRAINED or bound.template_available:
            return RepairTier.TEMPLATE_CONSTRAINED
        if requested in (None, RepairTier.DETERMINISTIC):
            return RepairTier.DETERMINISTIC
        return requested

    def admit_source_edit(
        self,
        operator: Mapping[str, Any] | AdmittedSourceEditOperator,
        *,
        predicted_path: str,
    ) -> AdmittedSourceEditOperator:
        """Admit one exact source edit through the existing operator contract."""

        path = _posix_path(predicted_path, "predicted_path")
        try:
            admitted = (
                operator
                if isinstance(operator, AdmittedSourceEditOperator)
                else AdmittedSourceEditOperator.from_mapping(operator)
            )
        except AdmittedSourceEditError as exc:
            raise RepairControllerError(str(exc) or "source_edit_operator_not_admitted") from exc
        if admitted.relative_path != path:
            raise RepairControllerError("source_edit_path_binding_mismatch")
        codes = classify_repair_path(admitted.relative_path, self._envelope.allowed_paths)
        if codes:
            raise RepairControllerError("+".join(codes))
        return admitted

    def evaluate(
        self,
        request: RepairControllerRequest | Mapping[str, Any],
        *,
        delegate: bool = False,
    ) -> RepairControllerResult:
        """Admit, select a tier, apply backoff, and compute merge disposition."""

        bound = RepairControllerRequest.from_mapping(request)
        rejection = self._rejection_codes(bound)
        operator_id = ""
        if bound.source_edit_operator is not None and "source_edit_not_admitted" not in rejection:
            try:
                admitted = self.admit_source_edit(
                    bound.source_edit_operator,
                    predicted_path=bound.predicted_files[0],
                )
                operator_id = admitted.operator_id
            except RepairControllerError as exc:
                text = str(exc)
                if "scope_escape" in text:
                    rejection.append("scope_escape")
                if "self_edit" in text:
                    rejection.append("self_edit")
                if "validator_policy_key_mutation" in text:
                    rejection.append("validator_policy_key_mutation")
                if "source_edit" in text or not any(
                    code in text
                    for code in ("scope_escape", "self_edit", "validator_policy_key_mutation")
                ):
                    rejection.append("source_edit_not_admitted")
        rejection = list(dict.fromkeys(rejection))
        if rejection:
            merge = evaluate_merge_conjunction(
                policy=self._policy,
                envelope=self._envelope,
                request=bound,
                rejection_codes=rejection,
                terminal_status=TerminalStatus.FAILED,
                receipt_authorizes_merge=False,
            )
            return RepairControllerResult(
                disposition=RepairControllerDisposition.REJECTED,
                repair_tier=bound.requested_tier,
                merge=merge,
                reason_codes=tuple(rejection),
                source_edit_operator_id=operator_id,
            )

        reused = self._reuse_identical_failure(bound)
        if reused is not None:
            status = (
                RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
                if reused.count >= self._max_identical
                else RepairControllerDisposition.IDENTICAL_FAILURE_REUSED
            )
            merge = evaluate_merge_conjunction(
                policy=self._policy,
                envelope=self._envelope,
                request=bound,
                rejection_codes=("identical_failure",),
                terminal_status=TerminalStatus.FAILED,
                receipt_authorizes_merge=False,
            )
            return RepairControllerResult(
                disposition=status,
                repair_tier=bound.requested_tier,
                merge=merge,
                reason_codes=(
                    status.value,
                    "diagnostic_reused",
                    "model_call_suppressed",
                ),
                diagnostic_reused=True,
                backoff_milliseconds=reused.backoff_milliseconds,
                model_call_count=0,
                failure_signature=reused.signature,
                diagnostic_receipt_id=reused.diagnostic_receipt_id,
                source_edit_operator_id=operator_id,
            )

        try:
            tier = self.select_tier(bound)
        except RepairControllerError as exc:
            code = "insufficient_context" if "context" in str(exc) else "tier_precondition"
            merge = evaluate_merge_conjunction(
                policy=self._policy,
                envelope=self._envelope,
                request=bound,
                rejection_codes=(code,),
                terminal_status=TerminalStatus.BLOCKED,
                receipt_authorizes_merge=False,
            )
            return RepairControllerResult(
                disposition=(
                    RepairControllerDisposition.INSUFFICIENT_CONTEXT
                    if code == "insufficient_context"
                    else RepairControllerDisposition.REJECTED
                ),
                repair_tier=bound.requested_tier,
                merge=merge,
                reason_codes=(code,),
            )

        plan = self._build_plan(bound, tier)
        model_calls = 0
        diagnostic_id = bound.diagnostic_receipt_id
        failure_signature = bound.failure_signature
        if tier is RepairTier.MODEL_ASSISTED_BOUNDED and self._model_call is not None:
            payload = self._model_call(request=bound, plan=plan)
            self._model_calls += 1
            model_calls = 1
            if isinstance(payload, Mapping):
                _reject_forbidden_keys(payload, "model_call result")
                diagnostic_id = str(payload.get("diagnostic_receipt_id") or diagnostic_id)
                failure_signature = str(payload.get("failure_signature") or failure_signature)

        engine_report = None
        disposition = RepairControllerDisposition.ADMITTED
        if delegate:
            engine_report = self._delegate(bound, plan)
            disposition = RepairControllerDisposition.ENGINE_DELEGATED

        succeeded = bool(bound.validation_receipt_ids)
        terminal = TerminalStatus.SUCCEEDED if succeeded else TerminalStatus.PENDING
        receipt = AutonomousRepairReceipt(
            plan_id=plan.plan_id,
            envelope_id=self._envelope.envelope_id,
            terminal_status=terminal,
            changed_paths=bound.changed_paths,
            validation_receipt_ids=bound.validation_receipt_ids,
            proof_receipt_ids=bound.proof_receipt_ids,
            adversarial_assurance_receipt_ids=bound.adversarial_assurance_receipt_ids,
            diagnostic_receipt_id=diagnostic_id,
            failure_signature=failure_signature,
            authorizes_merge=False,
        )
        merge = evaluate_merge_conjunction(
            policy=self._policy,
            envelope=self._envelope,
            request=bound,
            rejection_codes=(),
            terminal_status=terminal,
            receipt_authorizes_merge=receipt.authorizes_merge,
        )
        if failure_signature and terminal is not TerminalStatus.SUCCEEDED:
            self._record_failure(failure_signature, diagnostic_id)
        reasons = [disposition.value, tier.value, merge.disposition.value]
        if bound.suffix_receipt_id:
            reasons.append("suffix_bound")
        if operator_id:
            reasons.append("source_edit_admitted")
        return RepairControllerResult(
            disposition=disposition,
            repair_tier=tier,
            merge=merge,
            reason_codes=tuple(reasons),
            plan=plan,
            receipt=receipt,
            model_call_count=model_calls,
            engine_report=engine_report,
            source_edit_operator_id=operator_id,
            failure_signature=failure_signature,
            diagnostic_receipt_id=diagnostic_id,
        )

    def run(self, request: RepairControllerRequest | Mapping[str, Any]) -> RepairControllerResult:
        """Evaluate and delegate an admitted request to the existing engine."""

        return self.evaluate(request, delegate=True)

    def _rejection_codes(self, request: RepairControllerRequest) -> list[str]:
        codes: list[str] = []
        for path in request.scoped_paths:
            codes.extend(classify_repair_path(path, self._envelope.allowed_paths))
        if request.forbidden_symbols or self._forbidden_symbols:
            forbidden = set(request.forbidden_symbols) | set(self._forbidden_symbols)
            if forbidden.intersection(request.predicted_symbols):
                codes.append("forbidden_symbol")
        if self._envelope.allowed_symbols and not set(request.predicted_symbols).issubset(
            self._envelope.allowed_symbols
        ):
            codes.append("symbol_escape")
        if request.source_edit_operator is None and request.claimed_applied:
            codes.append("source_edit_not_admitted")
        return list(dict.fromkeys(codes))

    def _require_model_assisted_preconditions(self, request: RepairControllerRequest) -> None:
        if not request.predicted_files or not request.predicted_symbols:
            raise RepairControllerError("model-assisted repair requires exact files and symbols")
        if not request.context_reference_ids:
            raise RepairControllerError("model-assisted repair requires sufficient context")
        if not request.worktree_id:
            raise RepairControllerError("model-assisted repair requires an isolated worktree")
        tests = request.required_test_ids or self._envelope.required_test_ids
        if not tests:
            raise RepairControllerError("model-assisted repair requires predetermined tests")
        if not request.rollback_plan_id:
            raise RepairControllerError("model-assisted repair requires a rollback plan")

    def _build_plan(
        self, request: RepairControllerRequest, tier: RepairTier
    ) -> AutonomousRepairPlan:
        context_ids = request.context_reference_ids or (self._envelope.envelope_id,)
        tests = request.required_test_ids or self._envelope.required_test_ids
        proofs = request.required_proof_ids or self._envelope.required_proof_ids
        return AutonomousRepairPlan(
            objective_id=self._envelope.objective_id,
            task_id=self._envelope.task_id,
            repair_tier=tier,
            predicted_files=request.predicted_files,
            predicted_symbols=request.predicted_symbols,
            patch_envelope_id=self._envelope.envelope_id,
            context_reference_ids=context_ids,
            required_test_ids=tests,
            required_proof_ids=proofs,
            worktree_id=request.worktree_id,
            allowed_paths=self._envelope.allowed_paths,
            forbidden_symbols=request.forbidden_symbols or self._forbidden_symbols,
            rollback_plan_id=request.rollback_plan_id,
            risk_class=self._envelope.risk_assessment.risk_class,
            max_changed_files=request.max_changed_files,
            max_changed_lines=request.max_changed_lines,
        )

    def _reuse_identical_failure(
        self, request: RepairControllerRequest
    ) -> RepairFailureRecord | None:
        signature = request.failure_signature
        if not signature or request.new_evidence_since_failure:
            return None
        existing = self._failures.get(signature)
        if existing is None:
            return None
        count = existing.count + 1
        backoff = _backoff_milliseconds(
            count, base=self._base_backoff, maximum=self._max_backoff
        )
        reused = RepairFailureRecord(
            signature=signature,
            count=count,
            diagnostic_receipt_id=existing.diagnostic_receipt_id or request.diagnostic_receipt_id,
            backoff_milliseconds=backoff,
        )
        self._failures[signature] = reused
        return reused

    def _record_failure(self, signature: str, diagnostic_receipt_id: str) -> None:
        self._failures[signature] = RepairFailureRecord(
            signature=signature,
            count=1,
            diagnostic_receipt_id=diagnostic_receipt_id,
            backoff_milliseconds=_backoff_milliseconds(
                1, base=self._base_backoff, maximum=self._max_backoff
            ),
        )

    def _delegate(
        self, request: RepairControllerRequest, plan: AutonomousRepairPlan
    ) -> Mapping[str, Any]:
        if self._engine is None:
            raise RepairControllerError("existing AutonomousRepairEngine is required to delegate")
        if type(self._engine) is not AutonomousRepairEngine:
            raise RepairControllerError("engine must be the existing AutonomousRepairEngine")
        report = self._engine.run(
            [
                {
                    "work_id": request.work_id or plan.plan_id,
                    "operation": request.operation,
                    "path": request.predicted_files[0],
                    "symbol": request.predicted_symbols[0],
                    "write_paths": list(request.predicted_files),
                    "domain": "agent_supervisor",
                }
            ]
        )
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        payload["completion_authoritative"] = False
        payload["grants_execution_authority"] = False
        return MappingProxyType(payload)


__all__ = [
    "AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE",
    "AUTONOMOUS_REPAIR_CONTROLLER_SCHEMA",
    "DEFAULT_BASE_BACKOFF_MILLISECONDS",
    "DEFAULT_MAX_BACKOFF_MILLISECONDS",
    "DEFAULT_MAX_IDENTICAL_FAILURES",
    "LOW_RISK_MERGE_CONDITIONS",
    "PROTECTED_AUTHORITY_PATHS",
    "SELF_EDIT_PATHS",
    "VALIDATOR_POLICY_KEY_PATHS",
    "AutonomousRepairController",
    "RepairControllerDisposition",
    "RepairControllerError",
    "RepairControllerRequest",
    "RepairControllerResult",
    "RepairFailureRecord",
    "RepairMergeConjunction",
    "RepairMergeDisposition",
    "classify_repair_path",
    "evaluate_merge_conjunction",
    "path_escapes_allowed",
    "path_is_self_edit",
    "path_is_validator_policy_key",
]
