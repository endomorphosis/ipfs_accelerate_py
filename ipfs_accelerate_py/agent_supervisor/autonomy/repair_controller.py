# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded facade over the existing autonomous-repair engine.

``AutonomousRepairController@1`` selects among deterministic, template-
constrained, and model-assisted tiers, then binds one exact envelope,
isolated worktree, predetermined checks, repeated-failure backoff, and merge
disposition.  It is not a second repair engine, effect authority, or merge
authority.

Admission and execution stay with
``agent_supervisor.autonomous_repair.engine.AutonomousRepairEngine``,
the typed source-edit operator, and ``DecisionRuntime``.  Repair receipts
remain evidence: they never independently authorize merge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller@1"
)
REPAIR_CONTROLLER_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-request@1"
)
REPAIR_CONTROLLER_OUTCOME_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-outcome@1"
)
REPAIR_CONTROLLER_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-snapshot@1"
)
SOURCE_EDIT_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/source-edit-admission@1"
)
MAX_REPAIR_CONTROLLER_SNAPSHOT_BYTES: Final[int] = 4 * MAX_CANONICAL_RECORD_BYTES
DEFAULT_BASE_BACKOFF_MS: Final[int] = 100
DEFAULT_MAX_BACKOFF_MS: Final[int] = 10_000
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 3
SELF_EDIT_RELATIVE_PATH: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
)
ENGINE_PACKAGE_PREFIX: Final[str] = "ipfs_accelerate_py/agent_supervisor/autonomous_repair"

LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "autonomous_merge_enabled",
    "risk_at_most_r2",
    "reversible",
    "autonomy_allows_execute_reversible",
    "repair_succeeded",
    "current_validation_evidence",
    "required_tests_satisfied",
    "required_proofs_satisfied",
    "scope_contained",
    "protected_authority_unmodified",
    "self_edit_absent",
    "rollback_plan_bound",
    "isolated_worktree_bound",
    "predetermined_checks_complete",
)

_SELF_EDIT_PATHS: Final[frozenset[str]] = frozenset(
    {
        SELF_EDIT_RELATIVE_PATH,
        "test/api/autonomy/test_repair_controller.py",
    }
)
_PROTECTED_AUTHORITY_PREFIXES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py/agent_supervisor/validation/",
    "ipfs_accelerate_py/agent_supervisor/validation",
    "ipfs_accelerate_py/agent_supervisor/proof/",
    "ipfs_accelerate_py/agent_supervisor/proof",
    "ipfs_accelerate_py/agent_supervisor/verification/",
    "ipfs_accelerate_py/agent_supervisor/verification",
    f"{ENGINE_PACKAGE_PREFIX}/",
    ENGINE_PACKAGE_PREFIX,
    "config/",
    "config",
    "secrets/",
    "secrets",
    "credentials/",
    "credentials",
    ".ssh/",
    ".ssh",
)
_PROTECTED_AUTHORITY_SEGMENTS: Final[frozenset[str]] = frozenset(
    {
        "trusted_keys",
        "signing_keys",
        "private_keys",
        "verifier",
        "verifiers",
        "oracle",
        "oracles",
        "secrets",
        "credentials",
    }
)
_PROTECTED_AUTHORITY_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        "authorized_keys",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "policy.json",
        "policy.yaml",
        "policy.yml",
        "trusted_keys",
        "validator_policy.json",
    }
)
_PROTECTED_AUTHORITY_SUFFIXES: Final[tuple[str, ...]] = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
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
_MAX_SEQUENCE: Final[int] = (1 << 63) - 1


class RepairControllerError(ValueError):
    """Raised when repair-controller inputs themselves are malformed."""


class RepairControllerDisposition(str, Enum):
    """Closed outcome of one facade step.  None grants merge or effect authority."""

    ADMITTED = "admitted"
    EXECUTED = "executed"
    MERGE_ELIGIBLE = "merge_eligible"
    PROPOSAL_ONLY = "proposal_only"
    REJECTED_SCOPE_ESCAPE = "rejected_scope_escape"
    REJECTED_SELF_EDIT = "rejected_self_edit"
    REJECTED_PROTECTED_AUTHORITY = "rejected_protected_authority"
    REJECTED_MISSING_CHECKS = "rejected_missing_checks"
    REJECTED_MISSING_WORKTREE = "rejected_missing_worktree"
    REJECTED_MISSING_CONTEXT = "rejected_missing_context"
    REJECTED_SOURCE_EDIT = "rejected_source_edit"
    IDENTICAL_FAILURE_BACKOFF = "identical_failure_backoff"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    ROLLBACK = "rollback"
    BLOCKED = "blocked"
    FAILED = "failed"


class RepairMergeDisposition(str, Enum):
    """Merge recommendation.  Eligibility is never an authorization."""

    WITHHOLD = "withhold"
    PROPOSE = "propose"
    AUTONOMOUS_MERGE_ELIGIBLE = "autonomous_merge_eligible"


class SourceEditAdmissionDisposition(str, Enum):
    """Fail-closed source-edit admission.  Analysis rows cannot satisfy this."""

    ADMITTED_VALIDATION_PENDING = "admitted_validation_pending"
    REJECTED = "rejected"
    NOT_SOURCE_EDIT = "not_source_edit"


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise RepairControllerError(f"{name} must be a compact identifier")
    if required and not result:
        raise RepairControllerError(f"{name} is required")
    if result and (
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
    identifiers = _identifiers(value, name, required=required, preserve_order=True)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in identifiers:
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


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int = _MAX_SEQUENCE) -> int:
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


def path_under_prefix(path: str, prefix: str) -> bool:
    """Return True when *path* is exactly *prefix* or a descendant."""

    path_n = path.replace("\\", "/").strip()
    pref = prefix.replace("\\", "/").strip()
    if not pref:
        return False
    if pref.endswith("/"):
        pref = pref[:-1]
    return path_n == pref or path_n.startswith(pref + "/")


def path_under_any(path: str, prefixes: Sequence[str]) -> bool:
    return any(path_under_prefix(path, prefix) for prefix in prefixes)


def is_self_edit_path(path: str) -> bool:
    """Return True when a repair would edit this facade or its tests."""

    normalized = path.replace("\\", "/").strip()
    if normalized in _SELF_EDIT_PATHS:
        return True
    return PurePosixPath(normalized).name == "repair_controller.py" and path_under_prefix(
        normalized, "ipfs_accelerate_py/agent_supervisor/autonomy"
    )


def is_protected_authority_path(path: str) -> bool:
    """Return True for validator, policy, key, or self-protecting engine paths."""

    normalized = path.replace("\\", "/").strip().lower()
    if not normalized:
        return True
    if path_under_any(normalized, _PROTECTED_AUTHORITY_PREFIXES):
        return True
    parsed = PurePosixPath(normalized)
    if parsed.name in _PROTECTED_AUTHORITY_BASENAMES:
        return True
    if any(normalized.endswith(suffix) for suffix in _PROTECTED_AUTHORITY_SUFFIXES):
        return True
    if any(segment in _PROTECTED_AUTHORITY_SEGMENTS for segment in parsed.parts):
        return True
    collapsed = normalized.replace("-", "_")
    return "validator_policy" in collapsed or "policy_key" in collapsed


def scope_escape_paths(paths: Sequence[str], allowed: Sequence[str]) -> tuple[str, ...]:
    return tuple(path for path in paths if not path_under_any(path, allowed))


def select_repair_tier(
    *,
    requested: RepairTier | None,
    predicted_symbols: Sequence[str],
    context_reference_ids: Sequence[str],
    worktree_id: str,
    required_test_ids: Sequence[str],
    required_proof_ids: Sequence[str],
) -> RepairTier:
    """Software-first tier selection.  Model-assisted is never implicit."""

    if requested is not None:
        return _enum(requested, RepairTier, "requested_tier")
    if predicted_symbols:
        return RepairTier.TEMPLATE_CONSTRAINED
    if (
        context_reference_ids
        and worktree_id
        and (required_test_ids or required_proof_ids)
    ):
        return RepairTier.DETERMINISTIC
    return RepairTier.DETERMINISTIC


def _model_assisted_preconditions(
    *,
    predicted_files: Sequence[str],
    predicted_symbols: Sequence[str],
    context_reference_ids: Sequence[str],
    worktree_id: str,
    required_test_ids: Sequence[str],
    required_proof_ids: Sequence[str],
) -> tuple[str, ...]:
    missing: list[str] = []
    if not predicted_files:
        missing.append("exact_files_required")
    if not predicted_symbols:
        missing.append("exact_symbols_required")
    if not context_reference_ids:
        missing.append("sufficient_context_required")
    if not worktree_id:
        missing.append("isolated_worktree_required")
    if not required_test_ids and not required_proof_ids:
        missing.append("predetermined_checks_required")
    return tuple(missing)


def evaluate_low_risk_merge_conjunction(
    *,
    policy: AutonomyPolicy,
    envelope: AutonomyEnvelope,
    plan: AutonomousRepairPlan | None,
    receipt: AutonomousRepairReceipt | None,
    changed_paths: Sequence[str],
    validation_receipt_ids: Sequence[str],
    proof_receipt_ids: Sequence[str],
) -> Mapping[str, bool]:
    """Return every named low-risk merge condition.  Merge requires the conjunction."""

    risk = envelope.risk_assessment.risk_class
    required_tests = envelope.required_test_ids
    required_proofs = envelope.required_proof_ids
    if plan is not None:
        required_tests = tuple(sorted(set(required_tests) | set(plan.required_test_ids)))
        required_proofs = tuple(sorted(set(required_proofs) | set(plan.required_proof_ids)))
    allowed = envelope.allowed_paths
    if plan is not None:
        allowed = plan.allowed_paths
    succeeded = (
        receipt is not None and receipt.terminal_status is TerminalStatus.SUCCEEDED
    )
    validation_ids = tuple(validation_receipt_ids)
    if receipt is not None:
        validation_ids = tuple(sorted(set(validation_ids) | set(receipt.validation_receipt_ids)))
    proof_ids = tuple(proof_receipt_ids)
    if receipt is not None:
        proof_ids = tuple(sorted(set(proof_ids) | set(receipt.proof_receipt_ids)))
    inspected_paths = tuple(changed_paths)
    if receipt is not None and receipt.changed_paths:
        inspected_paths = receipt.changed_paths
    elif plan is not None and not inspected_paths:
        inspected_paths = plan.predicted_files
    conditions = {
        "autonomous_merge_enabled": policy.autonomous_merge_enabled is True,
        "risk_at_most_r2": risk.rank <= RiskClass.R2_REVERSIBLE_LOCAL.rank,
        "reversible": envelope.reversible is True and envelope.risk_assessment.reversible is True,
        "autonomy_allows_execute_reversible": policy.allows(
            AutonomyLevel.EXECUTE_REVERSIBLE, risk
        ),
        "repair_succeeded": succeeded,
        "current_validation_evidence": bool(validation_ids),
        "required_tests_satisfied": set(required_tests).issubset(validation_ids),
        "required_proofs_satisfied": set(required_proofs).issubset(proof_ids),
        "scope_contained": bool(inspected_paths)
        and not scope_escape_paths(inspected_paths, allowed)
        and not scope_escape_paths(inspected_paths, envelope.allowed_paths),
        "protected_authority_unmodified": not any(
            is_protected_authority_path(path) for path in inspected_paths
        ),
        "self_edit_absent": not any(is_self_edit_path(path) for path in inspected_paths),
        "rollback_plan_bound": bool(plan.rollback_plan_id) if plan is not None else False,
        "isolated_worktree_bound": bool(plan.worktree_id) if plan is not None else False,
        "predetermined_checks_complete": succeeded
        and set(required_tests).issubset(validation_ids)
        and set(required_proofs).issubset(proof_ids)
        and bool(validation_ids),
    }
    if set(conditions) != set(LOW_RISK_MERGE_CONDITIONS):
        raise RepairControllerError("low-risk merge conjunction is incomplete")
    return MappingProxyType({name: conditions[name] for name in LOW_RISK_MERGE_CONDITIONS})


def merge_disposition_for(
    conditions: Mapping[str, bool],
    *,
    risk: RiskClass,
) -> RepairMergeDisposition:
    if risk.rank >= RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE.rank:
        return RepairMergeDisposition.WITHHOLD
    if all(conditions[name] for name in LOW_RISK_MERGE_CONDITIONS):
        return RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    if risk is RiskClass.R3_BOUNDED_REPOSITORY_MUTATION and conditions.get(
        "repair_succeeded", False
    ):
        return RepairMergeDisposition.PROPOSE
    return RepairMergeDisposition.WITHHOLD


@dataclass(frozen=True)
class SourceEditAdmission:
    """Controller-level source-edit gate over the existing typed operator."""

    disposition: SourceEditAdmissionDisposition
    admitted: bool
    mutation_applied: bool
    relative_path: str = ""
    operator_id: str = ""
    old_digest: str = ""
    new_digest: str = ""
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, SourceEditAdmissionDisposition, "disposition"),
        )
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(
            self, "mutation_applied", _bool(self.mutation_applied, "mutation_applied")
        )
        object.__setattr__(
            self,
            "relative_path",
            _identifier(self.relative_path, "relative_path", required=False),
        )
        object.__setattr__(
            self, "operator_id", _identifier(self.operator_id, "operator_id", required=False)
        )
        object.__setattr__(
            self, "old_digest", _identifier(self.old_digest, "old_digest", required=False)
        )
        object.__setattr__(
            self, "new_digest", _identifier(self.new_digest, "new_digest", required=False)
        )
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", preserve_order=True),
        )
        if self.mutation_applied:
            raise RepairControllerError("repair controller cannot apply source mutations")
        if self.admitted and self.disposition is not (
            SourceEditAdmissionDisposition.ADMITTED_VALIDATION_PENDING
        ):
            raise RepairControllerError("admitted source edits remain validation-pending")
        if self.admitted is False and self.disposition is (
            SourceEditAdmissionDisposition.ADMITTED_VALIDATION_PENDING
        ):
            raise RepairControllerError("source-edit admission disposition is inconsistent")

    @property
    def authorizes_effect(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_EDIT_ADMISSION_SCHEMA,
            "disposition": self.disposition.value,
            "admitted": self.admitted,
            "mutation_applied": False,
            "relative_path": self.relative_path,
            "operator_id": self.operator_id,
            "old_digest": self.old_digest,
            "new_digest": self.new_digest,
            "reason_codes": list(self.reason_codes),
            "authorizes_effect": False,
            "policy_flag_is_source_edit_admission": False,
        }


@dataclass
class _FailureRecord:
    count: int
    diagnostic_receipt_id: str
    last_backoff_ms: int


@dataclass(frozen=True)
class RepairControllerRequest:
    """One facade invocation.  Open fields never carry prompts or source bodies."""

    predicted_files: tuple[str, ...] = ()
    predicted_symbols: tuple[str, ...] = ()
    requested_tier: RepairTier | None = None
    plan: AutonomousRepairPlan | None = None
    patch_envelope_id: str = ""
    context_reference_ids: tuple[str, ...] = ()
    required_test_ids: tuple[str, ...] = ()
    required_proof_ids: tuple[str, ...] = ()
    worktree_id: str = ""
    forbidden_symbols: tuple[str, ...] = ()
    rollback_plan_id: str = ""
    max_changed_files: int = 1
    max_changed_lines: int = 100
    changed_paths: tuple[str, ...] = ()
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""
    source_edit: Mapping[str, Any] | None = None
    work_items: tuple[Any, ...] = ()
    model_invoker: Callable[["RepairControllerRequest"], Any] | None = None
    now_ms: int = 0
    execute: bool = True
    new_evidence: bool = False
    rollback: bool = False
    allow_code_edit_materialize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_files",
            _posix_paths(self.predicted_files, "predicted_files", preserve_order=True),
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _identifiers(self.predicted_symbols, "predicted_symbols", preserve_order=True),
        )
        if self.requested_tier is not None:
            object.__setattr__(
                self,
                "requested_tier",
                _enum(self.requested_tier, RepairTier, "requested_tier"),
            )
        if self.plan is not None and not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan")
        object.__setattr__(
            self,
            "patch_envelope_id",
            _identifier(self.patch_envelope_id, "patch_envelope_id", required=False),
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
                _identifiers(getattr(self, name), name, preserve_order=True),
            )
        object.__setattr__(
            self, "worktree_id", _identifier(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self,
            "rollback_plan_id",
            _identifier(self.rollback_plan_id, "rollback_plan_id", required=False),
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
            "changed_paths",
            _posix_paths(self.changed_paths, "changed_paths", preserve_order=True),
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
        if self.source_edit is not None:
            if not isinstance(self.source_edit, Mapping):
                raise RepairControllerError("source_edit must be a mapping")
            if len(self.source_edit) > MAX_MAPPING_ITEMS:
                raise RepairControllerError("source_edit contains too many entries")
            _reject_forbidden_keys(self.source_edit, "source_edit")
            object.__setattr__(self, "source_edit", MappingProxyType(dict(self.source_edit)))
        if self.work_items is None:
            object.__setattr__(self, "work_items", ())
        elif isinstance(self.work_items, str) or not isinstance(self.work_items, Sequence):
            raise RepairControllerError("work_items must be a sequence")
        else:
            object.__setattr__(self, "work_items", tuple(self.work_items))
        if self.model_invoker is not None and not callable(self.model_invoker):
            raise RepairControllerError("model_invoker must be callable")
        object.__setattr__(self, "now_ms", _int(self.now_ms, "now_ms"))
        for name in ("execute", "new_evidence", "rollback", "allow_code_edit_materialize"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))


@dataclass(frozen=True)
class RepairControllerOutcome:
    """Autonomy-facing repair result.  Evidence only; never a merge permit."""

    disposition: RepairControllerDisposition
    selected_tier: RepairTier | None
    merge_disposition: RepairMergeDisposition
    low_risk_merge_conditions: Mapping[str, bool]
    reason_codes: tuple[str, ...]
    plan: AutonomousRepairPlan | None = None
    receipt: AutonomousRepairReceipt | None = None
    source_edit_admission: SourceEditAdmission | None = None
    model_call_count: int = 0
    engine_call_count: int = 0
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    engine_report_id: str = ""
    requires_decision_runtime: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairControllerDisposition, "disposition"),
        )
        if self.selected_tier is not None:
            object.__setattr__(
                self, "selected_tier", _enum(self.selected_tier, RepairTier, "selected_tier")
            )
        object.__setattr__(
            self,
            "merge_disposition",
            _enum(self.merge_disposition, RepairMergeDisposition, "merge_disposition"),
        )
        if not isinstance(self.low_risk_merge_conditions, Mapping):
            raise RepairControllerError("low_risk_merge_conditions must be a mapping")
        if set(self.low_risk_merge_conditions) != set(LOW_RISK_MERGE_CONDITIONS):
            raise RepairControllerError("low_risk_merge_conditions must report every named condition")
        frozen = {}
        for name in LOW_RISK_MERGE_CONDITIONS:
            frozen[name] = _bool(self.low_risk_merge_conditions[name], name)
        object.__setattr__(self, "low_risk_merge_conditions", MappingProxyType(frozen))
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", preserve_order=True),
        )
        if self.plan is not None and not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan")
        if self.receipt is not None and not isinstance(self.receipt, AutonomousRepairReceipt):
            raise RepairControllerError("receipt must be an AutonomousRepairReceipt")
        if self.source_edit_admission is not None and not isinstance(
            self.source_edit_admission, SourceEditAdmission
        ):
            raise RepairControllerError("source_edit_admission must be a SourceEditAdmission")
        object.__setattr__(self, "model_call_count", _int(self.model_call_count, "model_call_count"))
        object.__setattr__(
            self, "engine_call_count", _int(self.engine_call_count, "engine_call_count")
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
            "engine_report_id",
            _identifier(self.engine_report_id, "engine_report_id", required=False),
        )
        object.__setattr__(
            self,
            "requires_decision_runtime",
            _bool(self.requires_decision_runtime, "requires_decision_runtime"),
        )
        if self.receipt is not None and self.receipt.authorizes_merge:
            raise RepairControllerError("repair receipts cannot independently authorize merge")
        if (
            self.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
            and not all(self.low_risk_merge_conditions[name] for name in LOW_RISK_MERGE_CONDITIONS)
        ):
            raise RepairControllerError("autonomous merge requires the full low-risk conjunction")
        encoded = canonical_json(self.to_dict(include_identity=False)).encode("utf-8")
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise RepairControllerError("repair-controller outcome exceeds its bounded size")

    @property
    def outcome_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def authorizes_merge(self) -> bool:
        return False

    @property
    def authorizes_effect(self) -> bool:
        return False

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REPAIR_CONTROLLER_OUTCOME_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "engine_interface": AUTONOMOUS_REPAIR_INTERFACE,
            "disposition": self.disposition.value,
            "selected_tier": None if self.selected_tier is None else self.selected_tier.value,
            "merge_disposition": self.merge_disposition.value,
            "low_risk_merge_conditions": {
                name: bool(self.low_risk_merge_conditions[name])
                for name in LOW_RISK_MERGE_CONDITIONS
            },
            "reason_codes": list(self.reason_codes),
            "plan": None if self.plan is None else self.plan.to_dict(),
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "source_edit_admission": (
                None
                if self.source_edit_admission is None
                else self.source_edit_admission.to_dict()
            ),
            "model_call_count": self.model_call_count,
            "engine_call_count": self.engine_call_count,
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "engine_report_id": self.engine_report_id,
            "requires_decision_runtime": self.requires_decision_runtime,
            "authorizes_merge": False,
            "authorizes_effect": False,
        }
        if include_identity:
            payload["outcome_id"] = self.outcome_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepairControllerOutcome:
        if not isinstance(payload, Mapping):
            raise RepairControllerError("repair-controller outcome must be an object")
        expected = {
            "schema",
            "interface",
            "engine_interface",
            "disposition",
            "selected_tier",
            "merge_disposition",
            "low_risk_merge_conditions",
            "reason_codes",
            "plan",
            "receipt",
            "source_edit_admission",
            "model_call_count",
            "engine_call_count",
            "diagnostic_reused",
            "backoff_milliseconds",
            "engine_report_id",
            "requires_decision_runtime",
            "authorizes_merge",
            "authorizes_effect",
            "outcome_id",
        }
        extra = set(payload).difference(expected)
        if extra:
            raise RepairControllerError("repair-controller outcome contains unsupported fields")
        _reject_forbidden_keys(payload, "repair-controller outcome")
        if payload.get("schema") != REPAIR_CONTROLLER_OUTCOME_SCHEMA:
            raise RepairControllerError("unsupported repair-controller outcome schema")
        if payload.get("interface") != AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE:
            raise RepairControllerError("unsupported repair-controller interface")
        if payload.get("engine_interface") != AUTONOMOUS_REPAIR_INTERFACE:
            raise RepairControllerError("repair facade must keep AutonomousRepairEngine@1")
        if payload.get("authorizes_merge") is not False:
            raise RepairControllerError("repair-controller outcomes cannot authorize merge")
        if payload.get("authorizes_effect") is not False:
            raise RepairControllerError("repair-controller outcomes cannot authorize effects")
        plan = payload.get("plan")
        receipt = payload.get("receipt")
        admission = payload.get("source_edit_admission")
        result = cls(
            disposition=payload.get("disposition", ""),
            selected_tier=payload.get("selected_tier"),
            merge_disposition=payload.get("merge_disposition", ""),
            low_risk_merge_conditions=payload.get("low_risk_merge_conditions") or {},
            reason_codes=tuple(payload.get("reason_codes") or ()),
            plan=None if plan is None else AutonomousRepairPlan.from_dict(plan),
            receipt=None if receipt is None else AutonomousRepairReceipt.from_dict(receipt),
            source_edit_admission=(
                None
                if admission is None
                else SourceEditAdmission(
                    disposition=admission.get("disposition", ""),
                    admitted=admission.get("admitted"),
                    mutation_applied=admission.get("mutation_applied"),
                    relative_path=admission.get("relative_path", ""),
                    operator_id=admission.get("operator_id", ""),
                    old_digest=admission.get("old_digest", ""),
                    new_digest=admission.get("new_digest", ""),
                    reason_codes=tuple(admission.get("reason_codes") or ()),
                )
            ),
            model_call_count=payload.get("model_call_count", 0),
            engine_call_count=payload.get("engine_call_count", 0),
            diagnostic_reused=payload.get("diagnostic_reused"),
            backoff_milliseconds=payload.get("backoff_milliseconds", 0),
            engine_report_id=payload.get("engine_report_id", ""),
            requires_decision_runtime=payload.get("requires_decision_runtime"),
        )
        if payload.get("outcome_id") not in (None, "", result.outcome_id):
            raise RepairControllerError("repair-controller outcome identity does not match payload")
        return result


class AutonomousRepairController:
    """Tier-selection and scope facade over ``AutonomousRepairEngine``.

    The controller never implements repair operators, never writes source
    bytes, and never grants merge or effect authority.  ``DecisionRuntime``
    remains the mutation-permit boundary.
    """

    def __init__(
        self,
        *,
        envelope: AutonomyEnvelope,
        policy: AutonomyPolicy,
        engine: AutonomousRepairEngine | None = None,
        repo_root: str | Path | None = None,
        decision_runtime: Any = None,
        base_backoff_milliseconds: int = DEFAULT_BASE_BACKOFF_MS,
        max_backoff_milliseconds: int = DEFAULT_MAX_BACKOFF_MS,
        max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES,
    ) -> None:
        if not isinstance(envelope, AutonomyEnvelope):
            raise RepairControllerError("envelope must be an AutonomyEnvelope")
        if not isinstance(policy, AutonomyPolicy):
            raise RepairControllerError("policy must be an AutonomyPolicy")
        if envelope.policy_id != policy.policy_id:
            raise RepairControllerError("envelope policy_id does not match policy")
        if not policy.allows(envelope.autonomy_level, envelope.risk_assessment.risk_class):
            raise RepairControllerError("envelope autonomy level is not admitted by policy")
        if engine is not None and not isinstance(engine, AutonomousRepairEngine):
            raise RepairControllerError(
                "repair facade cannot substitute a second repair engine"
            )
        self._envelope = envelope
        self._policy = policy
        self._engine = engine
        self._repo_root = None if repo_root is None else Path(repo_root)
        self._decision_runtime = decision_runtime
        self._base_backoff_ms = _int(
            base_backoff_milliseconds, "base_backoff_milliseconds", minimum=1
        )
        self._max_backoff_ms = _int(
            max_backoff_milliseconds, "max_backoff_milliseconds", minimum=1
        )
        self._max_identical_failures = _int(
            max_identical_failures, "max_identical_failures", minimum=1, maximum=1_000
        )
        self._failures: dict[str, _FailureRecord] = {}
        self._model_call_count = 0
        self._engine_call_count = 0

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine_interface(self) -> str:
        return AUTONOMOUS_REPAIR_INTERFACE

    @property
    def engine_class(self) -> type[AutonomousRepairEngine]:
        return AutonomousRepairEngine

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
    def engine_call_count(self) -> int:
        return self._engine_call_count

    @property
    def authorizes_merge(self) -> bool:
        return False

    @property
    def authorizes_effect(self) -> bool:
        return False

    def _engine_instance(self) -> AutonomousRepairEngine:
        if self._engine is None:
            if self._repo_root is None:
                raise RepairControllerError("repair engine requires a repository root")
            self._engine = AutonomousRepairEngine(repo_root=self._repo_root)
        if not isinstance(self._engine, AutonomousRepairEngine):
            raise RepairControllerError(
                "repair facade cannot substitute a second repair engine"
            )
        return self._engine

    def _inspect_paths(self, paths: Sequence[str]) -> tuple[str, ...]:
        reasons: list[str] = []
        escaped = scope_escape_paths(paths, self._envelope.allowed_paths)
        if escaped:
            reasons.append("scope_escape")
        if any(is_self_edit_path(path) for path in paths):
            reasons.append("self_edit")
        if any(is_protected_authority_path(path) for path in paths):
            reasons.append("validator_policy_key")
        return tuple(reasons)

    def _rejection_disposition(self, reasons: Sequence[str]) -> RepairControllerDisposition:
        if "self_edit" in reasons:
            return RepairControllerDisposition.REJECTED_SELF_EDIT
        if "validator_policy_key" in reasons:
            return RepairControllerDisposition.REJECTED_PROTECTED_AUTHORITY
        if "scope_escape" in reasons:
            return RepairControllerDisposition.REJECTED_SCOPE_ESCAPE
        return RepairControllerDisposition.BLOCKED

    def _empty_conditions(self) -> Mapping[str, bool]:
        return MappingProxyType({name: False for name in LOW_RISK_MERGE_CONDITIONS})

    def _failure_key(self, request: RepairControllerRequest, plan: AutonomousRepairPlan) -> str:
        identity = request.failure_signature or request.diagnostic_receipt_id
        return content_identity(
            {
                "task_id": plan.task_id,
                "repair_tier": plan.repair_tier.value,
                "predicted_files": list(plan.predicted_files),
                "predicted_symbols": list(plan.predicted_symbols),
                "failure_identity": identity,
            }
        )

    def _backoff_ms(self, count: int) -> int:
        shift = max(0, count - 1)
        value = self._base_backoff_ms * (2**shift)
        return min(value, self._max_backoff_ms)

    def _bind_plan(self, request: RepairControllerRequest) -> AutonomousRepairPlan:
        if request.plan is not None:
            plan = request.plan
            if plan.objective_id != self._envelope.objective_id:
                raise RepairControllerError("repair plan objective_id does not match envelope")
            if plan.task_id != self._envelope.task_id:
                raise RepairControllerError("repair plan task_id does not match envelope")
            return plan
        files = request.predicted_files or self._envelope.allowed_paths[:1]
        symbols = request.predicted_symbols or self._envelope.allowed_symbols
        tests = request.required_test_ids or self._envelope.required_test_ids
        proofs = request.required_proof_ids or self._envelope.required_proof_ids
        context_ids = request.context_reference_ids
        worktree_id = request.worktree_id
        tier = select_repair_tier(
            requested=request.requested_tier,
            predicted_symbols=symbols,
            context_reference_ids=context_ids,
            worktree_id=worktree_id,
            required_test_ids=tests,
            required_proof_ids=proofs,
        )
        rollback_id = request.rollback_plan_id or f"rollback:{self._envelope.task_id}"
        patch_id = request.patch_envelope_id or self._envelope.envelope_id
        model_assisted = tier is RepairTier.MODEL_ASSISTED_BOUNDED
        if not worktree_id and not model_assisted:
            worktree_id = f"worktree:{self._envelope.task_id}"
        if not context_ids and not model_assisted:
            context_ids = (f"context:{self._envelope.tree_id}",)
        if not symbols:
            symbols = ("bounded_repair_target",)
        return AutonomousRepairPlan(
            objective_id=self._envelope.objective_id,
            task_id=self._envelope.task_id,
            repair_tier=tier,
            predicted_files=files,
            predicted_symbols=symbols,
            patch_envelope_id=patch_id,
            context_reference_ids=context_ids,
            required_test_ids=tests,
            required_proof_ids=proofs,
            worktree_id=worktree_id,
            allowed_paths=self._envelope.allowed_paths,
            forbidden_symbols=request.forbidden_symbols,
            rollback_plan_id=rollback_id,
            risk_class=self._envelope.risk_assessment.risk_class,
            max_changed_files=request.max_changed_files,
            max_changed_lines=request.max_changed_lines,
        )

    def admit_source_edit(
        self,
        source_edit: Mapping[str, Any] | None,
        *,
        predicted_files: Sequence[str],
        allow_code_edit_materialize: bool = False,
    ) -> SourceEditAdmission:
        """Admit a typed source-edit operator without applying it.

        A policy flag, catalog row, or body-free analysis plan is not
        source-edit admission.  The controller never writes bytes.
        """

        del allow_code_edit_materialize
        if source_edit is None:
            return SourceEditAdmission(
                disposition=SourceEditAdmissionDisposition.NOT_SOURCE_EDIT,
                admitted=False,
                mutation_applied=False,
                reason_codes=(
                    "source_edit_operator_missing",
                    "policy_flag_is_not_source_edit_admission",
                ),
            )
        try:
            operator = AdmittedSourceEditOperator.from_mapping(source_edit)
        except AdmittedSourceEditError as exc:
            return SourceEditAdmission(
                disposition=SourceEditAdmissionDisposition.REJECTED,
                admitted=False,
                mutation_applied=False,
                reason_codes=(str(exc), "typed_admitted_source_edit_operator_required"),
            )
        path = operator.relative_path.replace("\\", "/")
        reasons = list(self._inspect_paths((path,)))
        if predicted_files and path not in set(predicted_files) and not path_under_any(
            path, predicted_files
        ):
            reasons.append("source_edit_path_not_in_plan")
        if reasons:
            return SourceEditAdmission(
                disposition=SourceEditAdmissionDisposition.REJECTED,
                admitted=False,
                mutation_applied=False,
                relative_path=path,
                operator_id=operator.operator_id,
                old_digest=operator.old_digest,
                new_digest=operator.new_digest,
                reason_codes=tuple(dict.fromkeys(reasons)),
            )
        if self._repo_root is not None:
            try:
                operator.validate(
                    repo_root=self._repo_root.resolve(),
                    preferred_path=path,
                )
            except AdmittedSourceEditError as exc:
                return SourceEditAdmission(
                    disposition=SourceEditAdmissionDisposition.REJECTED,
                    admitted=False,
                    mutation_applied=False,
                    relative_path=path,
                    operator_id=operator.operator_id,
                    old_digest=operator.old_digest,
                    new_digest=operator.new_digest,
                    reason_codes=(str(exc),),
                )
        return SourceEditAdmission(
            disposition=SourceEditAdmissionDisposition.ADMITTED_VALIDATION_PENDING,
            admitted=True,
            mutation_applied=False,
            relative_path=path,
            operator_id=operator.operator_id,
            old_digest=operator.old_digest,
            new_digest=operator.new_digest,
            reason_codes=("typed_admitted_source_edit_operator", "validation_pending"),
        )

    def _outcome(
        self,
        *,
        disposition: RepairControllerDisposition,
        plan: AutonomousRepairPlan | None,
        receipt: AutonomousRepairReceipt | None,
        reason_codes: Sequence[str],
        source_edit_admission: SourceEditAdmission | None = None,
        diagnostic_reused: bool = False,
        backoff_milliseconds: int = 0,
        requires_decision_runtime: bool = False,
        changed_paths: Sequence[str] = (),
        validation_receipt_ids: Sequence[str] = (),
        proof_receipt_ids: Sequence[str] = (),
    ) -> RepairControllerOutcome:
        conditions = evaluate_low_risk_merge_conjunction(
            policy=self._policy,
            envelope=self._envelope,
            plan=plan,
            receipt=receipt,
            changed_paths=changed_paths,
            validation_receipt_ids=validation_receipt_ids,
            proof_receipt_ids=proof_receipt_ids,
        )
        merge = merge_disposition_for(
            conditions, risk=self._envelope.risk_assessment.risk_class
        )
        if (
            disposition is RepairControllerDisposition.EXECUTED
            and merge is RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
        ):
            disposition = RepairControllerDisposition.MERGE_ELIGIBLE
        elif (
            disposition is RepairControllerDisposition.EXECUTED
            and merge is RepairMergeDisposition.PROPOSE
        ):
            disposition = RepairControllerDisposition.PROPOSAL_ONLY
        return RepairControllerOutcome(
            disposition=disposition,
            selected_tier=None if plan is None else plan.repair_tier,
            merge_disposition=merge,
            low_risk_merge_conditions=conditions,
            reason_codes=tuple(reason_codes),
            plan=plan,
            receipt=receipt,
            source_edit_admission=source_edit_admission,
            model_call_count=self._model_call_count,
            engine_call_count=self._engine_call_count,
            diagnostic_reused=diagnostic_reused,
            backoff_milliseconds=backoff_milliseconds,
            requires_decision_runtime=requires_decision_runtime,
        )

    def _blocked(
        self,
        disposition: RepairControllerDisposition,
        plan: AutonomousRepairPlan | None,
        reasons: Sequence[str],
        *,
        source_edit_admission: SourceEditAdmission | None = None,
        diagnostic_reused: bool = False,
        backoff_milliseconds: int = 0,
        terminal: TerminalStatus = TerminalStatus.BLOCKED,
        failure_signature: str = "",
        diagnostic_receipt_id: str = "",
        changed_paths: Sequence[str] = (),
    ) -> RepairControllerOutcome:
        receipt = None
        if plan is not None:
            receipt = AutonomousRepairReceipt(
                plan_id=plan.plan_id,
                envelope_id=self._envelope.envelope_id,
                terminal_status=terminal,
                changed_paths=(),
                validation_receipt_ids=(),
                proof_receipt_ids=(),
                adversarial_assurance_receipt_ids=(),
                failure_signature=failure_signature,
                diagnostic_receipt_id=diagnostic_receipt_id,
                authorizes_merge=False,
            )
        return self._outcome(
            disposition=disposition,
            plan=plan,
            receipt=receipt,
            reason_codes=reasons,
            source_edit_admission=source_edit_admission,
            diagnostic_reused=diagnostic_reused,
            backoff_milliseconds=backoff_milliseconds,
            changed_paths=changed_paths,
        )

    def run(self, request: RepairControllerRequest | Mapping[str, Any]) -> RepairControllerOutcome:
        """Select a tier, enforce the envelope, and optionally compose the engine."""

        if not isinstance(request, RepairControllerRequest):
            if not isinstance(request, Mapping):
                raise RepairControllerError("repair request must be an object")
            request = RepairControllerRequest(**{
                key: value
                for key, value in request.items()
                if key != "schema"
            })
        try:
            plan = self._bind_plan(request)
        except RepairControllerError:
            raise
        except AutonomyContractError as exc:
            message = str(exc)
            if "isolated worktree" in message or "worktree_id is required" in message:
                return self._blocked(
                    RepairControllerDisposition.REJECTED_MISSING_WORKTREE,
                    None,
                    ("isolated_worktree_required",),
                )
            if "escapes allowed paths" in message or "exceed the patch envelope" in message:
                return self._blocked(
                    RepairControllerDisposition.REJECTED_SCOPE_ESCAPE,
                    None,
                    ("scope_escape",),
                )
            if "context_reference" in message:
                return self._blocked(
                    RepairControllerDisposition.REJECTED_MISSING_CONTEXT,
                    None,
                    ("sufficient_context_required",),
                )
            raise RepairControllerError(message) from exc
        except Exception as exc:
            raise RepairControllerError(str(exc)) from exc

        predicted = plan.predicted_files
        inspect_paths = predicted
        reasons = list(self._inspect_paths(inspect_paths))
        if scope_escape_paths(plan.allowed_paths, self._envelope.allowed_paths):
            reasons.append("scope_escape")
        if self._envelope.allowed_symbols and not set(plan.predicted_symbols).issubset(
            self._envelope.allowed_symbols
        ):
            reasons.append("scope_escape")
        if set(plan.predicted_symbols).intersection(plan.forbidden_symbols):
            reasons.append("forbidden_symbol")
        if plan.risk_class is not self._envelope.risk_assessment.risk_class:
            reasons.append("risk_mismatch")
        if reasons:
            return self._blocked(
                self._rejection_disposition(reasons),
                plan,
                tuple(dict.fromkeys(reasons)),
                changed_paths=inspect_paths,
            )

        if plan.repair_tier is RepairTier.MODEL_ASSISTED_BOUNDED:
            missing = _model_assisted_preconditions(
                predicted_files=plan.predicted_files,
                predicted_symbols=plan.predicted_symbols,
                context_reference_ids=plan.context_reference_ids,
                worktree_id=plan.worktree_id,
                required_test_ids=plan.required_test_ids,
                required_proof_ids=plan.required_proof_ids,
            )
            if missing:
                disposition = (
                    RepairControllerDisposition.REJECTED_MISSING_WORKTREE
                    if "isolated_worktree_required" in missing
                    else RepairControllerDisposition.REJECTED_MISSING_CONTEXT
                    if "sufficient_context_required" in missing
                    else RepairControllerDisposition.REJECTED_MISSING_CHECKS
                )
                return self._blocked(disposition, plan, missing)
            if self._envelope.autonomy_level.rank < AutonomyLevel.EXECUTE_REVERSIBLE.rank:
                return self._blocked(
                    RepairControllerDisposition.BLOCKED,
                    plan,
                    ("model_assisted_requires_reversible_or_isolated_autonomy",),
                )

        if request.rollback:
            receipt = AutonomousRepairReceipt(
                plan_id=plan.plan_id,
                envelope_id=self._envelope.envelope_id,
                terminal_status=TerminalStatus.CANCELLED,
                changed_paths=(),
                validation_receipt_ids=(),
                proof_receipt_ids=(),
                adversarial_assurance_receipt_ids=(),
                rollback_receipt_id=plan.rollback_plan_id,
                authorizes_merge=False,
            )
            return self._outcome(
                disposition=RepairControllerDisposition.ROLLBACK,
                plan=plan,
                receipt=receipt,
                reason_codes=("rollback_plan_bound", "isolated_worktree_discarded"),
            )

        failure_identity = request.failure_signature or request.diagnostic_receipt_id
        if failure_identity:
            key = self._failure_key(request, plan)
            record = self._failures.get(key)
            if record is not None and not request.new_evidence:
                record.count += 1
                backoff = self._backoff_ms(record.count)
                record.last_backoff_ms = backoff
                if record.count >= self._max_identical_failures:
                    return self._blocked(
                        RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED,
                        plan,
                        ("identical_failure_exhausted", "diagnosis_reused"),
                        diagnostic_reused=True,
                        backoff_milliseconds=backoff,
                        terminal=TerminalStatus.EXHAUSTED,
                        failure_signature=request.failure_signature,
                        diagnostic_receipt_id=record.diagnostic_receipt_id,
                    )
                return self._blocked(
                    RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF,
                    plan,
                    ("identical_failure_backoff", "diagnosis_reused", "model_call_suppressed"),
                    diagnostic_reused=True,
                    backoff_milliseconds=backoff,
                    terminal=TerminalStatus.BLOCKED,
                    failure_signature=request.failure_signature,
                    diagnostic_receipt_id=record.diagnostic_receipt_id,
                )

        if not request.execute:
            return self._outcome(
                disposition=RepairControllerDisposition.ADMITTED,
                plan=plan,
                receipt=None,
                reason_codes=("envelope_bound", plan.repair_tier.value),
                requires_decision_runtime=plan.repair_tier is RepairTier.MODEL_ASSISTED_BOUNDED,
            )

        source_admission: SourceEditAdmission | None = None
        if request.source_edit is not None or request.allow_code_edit_materialize:
            source_admission = self.admit_source_edit(
                request.source_edit,
                predicted_files=plan.predicted_files,
                allow_code_edit_materialize=request.allow_code_edit_materialize,
            )
            if not source_admission.admitted:
                return self._blocked(
                    RepairControllerDisposition.REJECTED_SOURCE_EDIT,
                    plan,
                    source_admission.reason_codes or ("source_edit_not_admitted",),
                    source_edit_admission=source_admission,
                )

        if request.work_items:
            engine = self._engine_instance()
            report = engine.run(request.work_items)
            self._engine_call_count += 1
            if int(getattr(report, "model_call_count", 0) or 0) != 0:
                return self._blocked(
                    RepairControllerDisposition.FAILED,
                    plan,
                    ("engine_model_call_forbidden",),
                    terminal=TerminalStatus.FAILED,
                )

        model_failed = False
        if plan.repair_tier is RepairTier.MODEL_ASSISTED_BOUNDED:
            if request.model_invoker is None:
                return self._blocked(
                    RepairControllerDisposition.BLOCKED,
                    plan,
                    ("model_invoker_required_for_model_assisted_tier",),
                    requires_decision_runtime=True,
                )
            self._model_call_count += 1
            try:
                invoked = request.model_invoker(request)
            except Exception:
                model_failed = True
                invoked = False
            if invoked is False:
                model_failed = True
            elif isinstance(invoked, Mapping) and invoked.get("ok") is False:
                model_failed = True

        changed_paths = request.changed_paths or (
            plan.predicted_files if source_admission is not None and source_admission.admitted else ()
        )
        changed_reasons = list(self._inspect_paths(changed_paths)) if changed_paths else []
        if changed_reasons:
            return self._blocked(
                self._rejection_disposition(changed_reasons),
                plan,
                tuple(dict.fromkeys(changed_reasons)),
                source_edit_admission=source_admission,
                changed_paths=changed_paths,
            )

        if model_failed or request.failure_signature:
            if failure_identity:
                key = self._failure_key(request, plan)
                previous = self._failures.get(key)
                count = 1 if previous is None else previous.count + 1
                diagnostic = request.diagnostic_receipt_id or (
                    previous.diagnostic_receipt_id if previous is not None else ""
                )
                self._failures[key] = _FailureRecord(
                    count=count,
                    diagnostic_receipt_id=diagnostic,
                    last_backoff_ms=self._backoff_ms(count),
                )
            if model_failed or not request.validation_receipt_ids:
                return self._blocked(
                    RepairControllerDisposition.FAILED,
                    plan,
                    ("repair_failed", "rollback_available"),
                    source_edit_admission=source_admission,
                    terminal=TerminalStatus.FAILED,
                    failure_signature=request.failure_signature,
                    diagnostic_receipt_id=request.diagnostic_receipt_id,
                    changed_paths=changed_paths,
                )

        validation_ids = request.validation_receipt_ids
        proof_ids = request.proof_receipt_ids
        required_tests = tuple(
            sorted(set(self._envelope.required_test_ids) | set(plan.required_test_ids))
        )
        required_proofs = tuple(
            sorted(set(self._envelope.required_proof_ids) | set(plan.required_proof_ids))
        )
        checks_complete = set(required_tests).issubset(validation_ids) and set(
            required_proofs
        ).issubset(proof_ids)
        if required_tests and not checks_complete:
            terminal = TerminalStatus.BLOCKED
            disposition = RepairControllerDisposition.REJECTED_MISSING_CHECKS
            reasons = ("predetermined_checks_incomplete",)
        elif validation_ids and checks_complete:
            terminal = TerminalStatus.SUCCEEDED
            disposition = RepairControllerDisposition.EXECUTED
            reasons = ("predetermined_checks_complete", plan.repair_tier.value)
        elif source_admission is not None and source_admission.admitted:
            terminal = TerminalStatus.BLOCKED
            disposition = RepairControllerDisposition.ADMITTED
            reasons = ("source_edit_validation_pending",)
        else:
            terminal = TerminalStatus.BLOCKED
            disposition = RepairControllerDisposition.ADMITTED
            reasons = ("envelope_bound", "validation_pending")

        receipt = AutonomousRepairReceipt(
            plan_id=plan.plan_id,
            envelope_id=self._envelope.envelope_id,
            terminal_status=terminal,
            changed_paths=changed_paths if terminal is TerminalStatus.SUCCEEDED else (),
            validation_receipt_ids=validation_ids if terminal is TerminalStatus.SUCCEEDED else (),
            proof_receipt_ids=proof_ids if terminal is TerminalStatus.SUCCEEDED else (),
            adversarial_assurance_receipt_ids=(
                request.adversarial_assurance_receipt_ids
                if terminal is TerminalStatus.SUCCEEDED
                else ()
            ),
            rollback_receipt_id=plan.rollback_plan_id if terminal is TerminalStatus.SUCCEEDED else "",
            authorizes_merge=False,
        )
        return self._outcome(
            disposition=disposition,
            plan=plan,
            receipt=receipt,
            reason_codes=reasons,
            source_edit_admission=source_admission,
            requires_decision_runtime=plan.repair_tier is RepairTier.MODEL_ASSISTED_BOUNDED
            or self._decision_runtime is not None,
            changed_paths=changed_paths if terminal is TerminalStatus.SUCCEEDED else (),
            validation_receipt_ids=validation_ids if terminal is TerminalStatus.SUCCEEDED else (),
            proof_receipt_ids=proof_ids if terminal is TerminalStatus.SUCCEEDED else (),
        )

    def snapshot(self) -> Mapping[str, Any]:
        payload = {
            "schema": REPAIR_CONTROLLER_SNAPSHOT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "engine_interface": AUTONOMOUS_REPAIR_INTERFACE,
            "envelope_id": self._envelope.envelope_id,
            "policy_id": self._policy.policy_id,
            "model_call_count": self._model_call_count,
            "engine_call_count": self._engine_call_count,
            "failures": {
                key: {
                    "count": record.count,
                    "diagnostic_receipt_id": record.diagnostic_receipt_id,
                    "last_backoff_ms": record.last_backoff_ms,
                }
                for key, record in sorted(self._failures.items())
            },
            "authorizes_merge": False,
            "authorizes_effect": False,
        }
        payload["snapshot_id"] = content_identity(payload)
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > MAX_REPAIR_CONTROLLER_SNAPSHOT_BYTES:
            raise RepairControllerError("repair-controller snapshot exceeds its bounded size")
        return MappingProxyType(payload)


__all__ = [
    "AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE",
    "AUTONOMOUS_REPAIR_CONTROLLER_SCHEMA",
    "ENGINE_PACKAGE_PREFIX",
    "LOW_RISK_MERGE_CONDITIONS",
    "REPAIR_CONTROLLER_OUTCOME_SCHEMA",
    "REPAIR_CONTROLLER_REQUEST_SCHEMA",
    "REPAIR_CONTROLLER_SNAPSHOT_SCHEMA",
    "SELF_EDIT_RELATIVE_PATH",
    "SOURCE_EDIT_ADMISSION_SCHEMA",
    "AutonomousRepairController",
    "RepairControllerDisposition",
    "RepairControllerError",
    "RepairControllerOutcome",
    "RepairControllerRequest",
    "RepairMergeDisposition",
    "SourceEditAdmission",
    "SourceEditAdmissionDisposition",
    "evaluate_low_risk_merge_conjunction",
    "is_protected_authority_path",
    "is_self_edit_path",
    "merge_disposition_for",
    "path_under_any",
    "path_under_prefix",
    "scope_escape_paths",
    "select_repair_tier",
]
