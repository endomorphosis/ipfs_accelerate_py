# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded facade over the existing autonomous-repair engine.

``AutonomousRepairController@1`` selects a repair tier, binds one exact
envelope, and refuses scope escape, self-edit, and validator/policy/key
mutation.  It is not a second repair engine, executor, merge authority, or
``DecisionRuntime``.  Admission and execution stay with
``agent_supervisor.autonomous_repair`` and
``agent_supervisor.context.decision_runtime.DecisionRuntime``.

Model-assisted repair additionally requires exact files and symbols, a bounded
patch envelope, sufficient context, an isolated worktree, predetermined
tests/proofs, and protected authority paths.  Identical failures reuse the
prior diagnosis and back off instead of repeating a model call.  Autonomous
merge is a conjunction of every stated low-risk condition and never applies
to R3-or-higher work, which remains proposal-only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..autonomous_repair.contracts import (
    AUTONOMOUS_REPAIR_INTERFACE,
    AutonomousRepairReport,
    RepairWorkItem,
)
from ..autonomous_repair.engine import AutonomousRepairEngine
from ..autonomous_repair.materialize import (
    AdmittedSourceEditError,
    AdmittedSourceEditOperator,
    AutonomousRepairMaterializer,
)
from ..context.decision_runtime import DecisionRuntime
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
from .receding_horizon import (
    PlanSuffixInvalidationReceipt,
    RecedingHorizonDisposition,
)

AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE: Final[str] = "AutonomousRepairController@1"
AUTONOMOUS_REPAIR_CONTROLLER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller@1"
)
REPAIR_CONTROLLER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-result@1"
)
REPAIR_CONTROLLER_MODULE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
)
AUTONOMOUS_REPAIR_PACKAGE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair"
)
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 3
DEFAULT_BASE_BACKOFF_MILLISECONDS: Final[int] = 10
DEFAULT_MAX_CHANGED_FILES: Final[int] = 8
DEFAULT_MAX_CHANGED_LINES: Final[int] = 400
MAX_REPAIR_CONTROLLER_RESULT_BYTES: Final[int] = 4 * MAX_CANONICAL_RECORD_BYTES

SELF_EDIT_PATHS: Final[tuple[str, ...]] = (
    REPAIR_CONTROLLER_MODULE_REL,
    AUTONOMOUS_REPAIR_PACKAGE_REL,
)

PROTECTED_AUTHORITY_PATHS: Final[tuple[str, ...]] = (
    REPAIR_CONTROLLER_MODULE_REL,
    AUTONOMOUS_REPAIR_PACKAGE_REL,
    "ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",
    "ipfs_accelerate_py/agent_supervisor/validation",
    "ipfs_accelerate_py/agent_supervisor/proof",
    "ipfs_accelerate_py/llm_router.py",
    "config/agent_supervisor_autonomous_meta_controller_scheduler.json",
)

_BLOCKED_AUTHORITY_SEGMENTS: Final[frozenset[str]] = frozenset(
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
_BLOCKED_AUTHORITY_SUFFIXES: Final[tuple[str, ...]] = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
)
_BLOCKED_AUTHORITY_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        "authorized_keys",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "oracle.json",
        "policy.json",
        "policy.yaml",
        "policy.yml",
        "trusted_keys.json",
        "validator-policy.json",
        "validator_policy.json",
        "validator_policy_key.json",
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
_SUFFIX_BACKOFF_DISPOSITIONS: Final[frozenset[RecedingHorizonDisposition]] = frozenset(
    {
        RecedingHorizonDisposition.UNCHANGED_BACKOFF,
        RecedingHorizonDisposition.IDENTICAL_FAILURE_EXHAUSTED,
        RecedingHorizonDisposition.RETRY_BUDGET_EXHAUSTED,
        RecedingHorizonDisposition.FAILURE_MEMORY_BOUND_REACHED,
        RecedingHorizonDisposition.REPAIR_BOUND_EXCEEDED,
    }
)

# Closed conjunction for R2 autonomous merge.  Every named condition must hold.
LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "autonomous_merge_enabled",
    "risk_class_r2_reversible_local",
    "envelope_reversible",
    "risk_assessment_reversible",
    "policy_allows_execute_reversible",
    "autonomy_level_allows_execution",
    "isolated_worktree",
    "exact_paths_within_envelope",
    "exact_symbols_bound",
    "no_protected_authority_paths",
    "no_forbidden_symbols",
    "predetermined_tests_current",
    "predetermined_proofs_current",
    "rollback_plan_bound",
    "repair_succeeded",
    "validation_evidence_current",
    "changed_paths_within_predicted",
    "envelope_identity_bound",
    "adversarial_assurance_current",
    "receipt_does_not_authorize_merge",
)


class RepairControllerError(ValueError):
    """Raised when repair-controller inputs themselves are malformed."""


class RepairControllerDisposition(str, Enum):
    """Closed outcome of one facade step.  None grants merge or effect authority."""

    ADMITTED = "admitted"
    ENGINE_DELEGATED = "engine_delegated"
    REJECTED_SCOPE_ESCAPE = "rejected_scope_escape"
    REJECTED_SELF_EDIT = "rejected_self_edit"
    REJECTED_VALIDATOR_POLICY_KEY = "rejected_validator_policy_key"
    REJECTED_MISSING_WORKTREE = "rejected_missing_worktree"
    REJECTED_MISSING_CHECKS = "rejected_missing_checks"
    REJECTED_INSUFFICIENT_CONTEXT = "rejected_insufficient_context"
    REJECTED_MISSING_ENVELOPE = "rejected_missing_envelope"
    REJECTED_SOURCE_EDIT = "rejected_source_edit"
    IDENTICAL_FAILURE_BACKOFF = "identical_failure_backoff"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    PROPOSAL_ONLY = "proposal_only"
    MERGE_ELIGIBLE = "merge_eligible"
    ROLLED_BACK = "rolled_back"


class MergeDisposition(str, Enum):
    """Closed merge recommendation.  Evidence only; never a merge permit."""

    NOT_ELIGIBLE = "not_eligible"
    AUTONOMOUS_MERGE_ELIGIBLE = "autonomous_merge_eligible"
    PROPOSAL_ONLY = "proposal_only"
    REJECTED = "rejected"


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
    if not isinstance(value, str) or not value.strip():
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    result = value.strip().replace("\\", "/")
    parsed = PurePosixPath(result)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or result in {".", ""}
        or "\x00" in result
    ):
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    return parsed.as_posix()


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


def path_under_prefix(path: str, prefix: str) -> bool:
    """Return True when *path* is *prefix* or a descendant of it."""

    path_n = path.replace("\\", "/").strip("/")
    pref = prefix.replace("\\", "/").strip("/")
    if not pref:
        return False
    return path_n == pref or path_n.startswith(pref + "/")


def path_under_any(path: str, prefixes: Sequence[str]) -> bool:
    return any(path_under_prefix(path, prefix) for prefix in prefixes)


def is_self_edit_path(path: str) -> bool:
    """Return True when *path* would edit this facade or the repair engine."""

    return path_under_any(path, SELF_EDIT_PATHS)


def is_validator_policy_key_path(path: str) -> bool:
    """Return True when *path* is a validator, policy, or trusted-key surface."""

    lower = path.replace("\\", "/").strip("/").lower()
    parsed = PurePosixPath(lower)
    if parsed.name in _BLOCKED_AUTHORITY_BASENAMES:
        return True
    if any(lower.endswith(suffix) for suffix in _BLOCKED_AUTHORITY_SUFFIXES):
        return True
    if any(segment in _BLOCKED_AUTHORITY_SEGMENTS for segment in parsed.parts):
        return True
    if "validator-policy" in lower or "validator_policy" in lower:
        return True
    return False


def is_protected_authority_path(path: str, extra: Sequence[str] = ()) -> bool:
    """Return True when *path* is a self-protecting or authority surface."""

    if path_under_any(path, (*PROTECTED_AUTHORITY_PATHS, *extra)):
        return True
    return is_self_edit_path(path) or is_validator_policy_key_path(path)


def classify_forbidden_path(
    path: str,
    *,
    allowed_paths: Sequence[str],
    extra_protected: Sequence[str] = (),
) -> str | None:
    """Return a closed rejection code, or None when the path is in envelope."""

    if is_self_edit_path(path):
        return "self_edit"
    if is_validator_policy_key_path(path) or path_under_any(path, extra_protected):
        return "validator_policy_key"
    if is_protected_authority_path(path, extra_protected):
        return "protected_authority_path"
    if allowed_paths and not path_under_any(path, allowed_paths):
        return "scope_escape"
    return None


@dataclass(frozen=True)
class FailureMemoryRecord:
    """Bounded identical-failure memory.  Stores no prompts or transcripts."""

    failure_signature: str
    diagnostic_receipt_id: str
    attempts: int
    backoff_milliseconds: int
    last_tier: RepairTier

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
        object.__setattr__(self, "attempts", _int(self.attempts, "attempts", minimum=1))
        object.__setattr__(
            self,
            "backoff_milliseconds",
            _int(self.backoff_milliseconds, "backoff_milliseconds"),
        )
        object.__setattr__(self, "last_tier", _enum(self.last_tier, RepairTier, "last_tier"))


@dataclass(frozen=True)
class RepairRequest:
    """One bounded repair request.  Never a prompt, transcript, or source body."""

    envelope: AutonomyEnvelope
    policy: AutonomyPolicy
    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    requested_tier: RepairTier = RepairTier.DETERMINISTIC
    plan: AutonomousRepairPlan | None = None
    context_reference_ids: tuple[str, ...] = ()
    required_test_ids: tuple[str, ...] = ()
    required_proof_ids: tuple[str, ...] = ()
    worktree_id: str = ""
    rollback_plan_id: str = ""
    forbidden_symbols: tuple[str, ...] = ()
    max_changed_files: int = DEFAULT_MAX_CHANGED_FILES
    max_changed_lines: int = DEFAULT_MAX_CHANGED_LINES
    context_sufficient: bool = False
    isolated_worktree: bool = False
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""
    source_edit_operator: Mapping[str, Any] | None = None
    work_items: tuple[RepairWorkItem | Mapping[str, Any], ...] = ()
    suffix_receipt: PlanSuffixInvalidationReceipt | None = None
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    extra_protected_paths: tuple[str, ...] = ()
    execute: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, AutonomyEnvelope):
            raise RepairControllerError("envelope must be an AutonomyEnvelope")
        if not isinstance(self.policy, AutonomyPolicy):
            raise RepairControllerError("policy must be an AutonomyPolicy")
        if self.envelope.policy_id != self.policy.policy_id:
            raise RepairControllerError("envelope policy_id does not match policy")
        object.__setattr__(
            self, "predicted_files", _paths(self.predicted_files, "predicted_files", required=True)
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _identifiers(self.predicted_symbols, "predicted_symbols", required=True),
        )
        object.__setattr__(
            self, "requested_tier", _enum(self.requested_tier, RepairTier, "requested_tier")
        )
        if self.plan is not None and not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan or None")
        object.__setattr__(
            self,
            "context_reference_ids",
            _identifiers(self.context_reference_ids, "context_reference_ids"),
        )
        object.__setattr__(
            self, "required_test_ids", _identifiers(self.required_test_ids, "required_test_ids")
        )
        object.__setattr__(
            self, "required_proof_ids", _identifiers(self.required_proof_ids, "required_proof_ids")
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
            self, "forbidden_symbols", _identifiers(self.forbidden_symbols, "forbidden_symbols")
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
            self, "context_sufficient", _bool(self.context_sufficient, "context_sufficient")
        )
        object.__setattr__(
            self, "isolated_worktree", _bool(self.isolated_worktree, "isolated_worktree")
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
        if self.source_edit_operator is not None:
            if not isinstance(self.source_edit_operator, Mapping):
                raise RepairControllerError("source_edit_operator must be a mapping")
            _reject_forbidden_keys(self.source_edit_operator, "source_edit_operator")
            object.__setattr__(
                self, "source_edit_operator", MappingProxyType(dict(self.source_edit_operator))
            )
        items = self.work_items or ()
        if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
            raise RepairControllerError("work_items must be a sequence")
        if len(items) > MAX_SEQUENCE_ITEMS:
            raise RepairControllerError("work_items contains too many items")
        object.__setattr__(self, "work_items", tuple(items))
        if self.suffix_receipt is not None and not isinstance(
            self.suffix_receipt, PlanSuffixInvalidationReceipt
        ):
            raise RepairControllerError("suffix_receipt must be a PlanSuffixInvalidationReceipt")
        object.__setattr__(
            self,
            "validation_receipt_ids",
            _identifiers(self.validation_receipt_ids, "validation_receipt_ids"),
        )
        object.__setattr__(
            self, "proof_receipt_ids", _identifiers(self.proof_receipt_ids, "proof_receipt_ids")
        )
        object.__setattr__(
            self,
            "adversarial_assurance_receipt_ids",
            _identifiers(
                self.adversarial_assurance_receipt_ids, "adversarial_assurance_receipt_ids"
            ),
        )
        object.__setattr__(self, "changed_paths", _paths(self.changed_paths, "changed_paths"))
        object.__setattr__(
            self,
            "extra_protected_paths",
            _paths(self.extra_protected_paths, "extra_protected_paths"),
        )
        object.__setattr__(self, "execute", _bool(self.execute, "execute"))
        if len(self.predicted_files) > self.max_changed_files:
            raise RepairControllerError("predicted repair files exceed the patch envelope")

    @property
    def allowed_paths(self) -> tuple[str, ...]:
        if self.plan is not None:
            return self.plan.allowed_paths
        return self.envelope.allowed_paths

    @property
    def effective_worktree_id(self) -> str:
        if self.plan is not None:
            return self.plan.worktree_id
        return self.worktree_id

    @property
    def effective_rollback_plan_id(self) -> str:
        if self.plan is not None:
            return self.plan.rollback_plan_id
        return self.rollback_plan_id

    @property
    def candidate_paths(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.predicted_files) | set(self.changed_paths)))

    def failure_key(self) -> str:
        material = {
            "envelope_id": self.envelope.envelope_id,
            "failure_signature": self.failure_signature,
            "predicted_files": list(self.predicted_files),
            "predicted_symbols": list(self.predicted_symbols),
        }
        return content_identity(material)


@dataclass(frozen=True)
class RepairControllerResult:
    """Autonomy-facing receipt.  Evidence only; never a merge or effect permit."""

    disposition: RepairControllerDisposition
    selected_tier: RepairTier
    reason_codes: tuple[str, ...]
    plan: AutonomousRepairPlan | None = None
    receipt: AutonomousRepairReceipt | None = None
    merge_disposition: MergeDisposition = MergeDisposition.NOT_ELIGIBLE
    satisfied_merge_conditions: tuple[str, ...] = ()
    missing_merge_conditions: tuple[str, ...] = ()
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    model_call_count: int = 0
    engine_report: Mapping[str, Any] | None = None
    engine_interface: str = AUTONOMOUS_REPAIR_INTERFACE
    source_edit_admitted: bool = False
    materializer_invoked: bool = False
    rolled_back: bool = False

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
            _identifiers(self.reason_codes, "reason_codes", preserve_order=True),
        )
        if self.plan is not None and not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan or None")
        if self.receipt is not None and not isinstance(self.receipt, AutonomousRepairReceipt):
            raise RepairControllerError("receipt must be an AutonomousRepairReceipt or None")
        object.__setattr__(
            self,
            "merge_disposition",
            _enum(self.merge_disposition, MergeDisposition, "merge_disposition"),
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
                self.missing_merge_conditions, "missing_merge_conditions", preserve_order=True
            ),
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
            self, "model_call_count", _int(self.model_call_count, "model_call_count")
        )
        if self.engine_report is not None:
            if not isinstance(self.engine_report, Mapping):
                raise RepairControllerError("engine_report must be a mapping")
            _reject_forbidden_keys(self.engine_report, "engine_report")
            object.__setattr__(self, "engine_report", MappingProxyType(dict(self.engine_report)))
        object.__setattr__(
            self, "engine_interface", _identifier(self.engine_interface, "engine_interface")
        )
        if self.engine_interface != AUTONOMOUS_REPAIR_INTERFACE:
            raise RepairControllerError("repair controller cannot bind a second repair engine")
        for name in ("source_edit_admitted", "materializer_invoked", "rolled_back"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        encoded = canonical_json(self.to_dict(include_identity=False)).encode("utf-8")
        if len(encoded) > MAX_REPAIR_CONTROLLER_RESULT_BYTES:
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

    @property
    def rejected(self) -> bool:
        return self.disposition.value.startswith("rejected_")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": REPAIR_CONTROLLER_RESULT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "disposition": self.disposition.value,
            "selected_tier": self.selected_tier.value,
            "reason_codes": list(self.reason_codes),
            "plan": None if self.plan is None else self.plan.to_dict(),
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "merge_disposition": self.merge_disposition.value,
            "satisfied_merge_conditions": list(self.satisfied_merge_conditions),
            "missing_merge_conditions": list(self.missing_merge_conditions),
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "model_call_count": self.model_call_count,
            "engine_report": None if self.engine_report is None else dict(self.engine_report),
            "engine_interface": self.engine_interface,
            "source_edit_admitted": self.source_edit_admitted,
            "materializer_invoked": self.materializer_invoked,
            "rolled_back": self.rolled_back,
            "authorizes_effect": False,
            "authorizes_merge": False,
        }
        if include_identity:
            payload["result_id"] = self.result_id
        return payload


def evaluate_low_risk_merge_conditions(
    *,
    policy: AutonomyPolicy,
    envelope: AutonomyEnvelope,
    plan: AutonomousRepairPlan,
    receipt: AutonomousRepairReceipt | None,
    isolated_worktree: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (satisfied, missing) for the closed R2 merge conjunction."""

    checks: dict[str, bool] = {
        "autonomous_merge_enabled": policy.autonomous_merge_enabled is True,
        "risk_class_r2_reversible_local": plan.risk_class is RiskClass.R2_REVERSIBLE_LOCAL
        and envelope.risk_assessment.risk_class is RiskClass.R2_REVERSIBLE_LOCAL,
        "envelope_reversible": envelope.reversible is True,
        "risk_assessment_reversible": envelope.risk_assessment.reversible is True,
        "policy_allows_execute_reversible": policy.allows(
            AutonomyLevel.EXECUTE_REVERSIBLE, plan.risk_class
        ),
        "autonomy_level_allows_execution": envelope.autonomy_level.rank
        >= AutonomyLevel.EXECUTE_REVERSIBLE.rank,
        "isolated_worktree": isolated_worktree and bool(plan.worktree_id),
        "exact_paths_within_envelope": bool(plan.predicted_files)
        and all(path_under_any(path, envelope.allowed_paths) for path in plan.predicted_files)
        and all(path_under_any(path, plan.allowed_paths) for path in plan.predicted_files),
        "exact_symbols_bound": bool(plan.predicted_symbols),
        "no_protected_authority_paths": not any(
            is_protected_authority_path(path) for path in plan.predicted_files
        ),
        "no_forbidden_symbols": not any(
            symbol in set(plan.forbidden_symbols) for symbol in plan.predicted_symbols
        ),
        "predetermined_tests_current": bool(plan.required_test_ids)
        and receipt is not None
        and set(plan.required_test_ids).issubset(set(receipt.validation_receipt_ids)),
        "predetermined_proofs_current": (
            not plan.required_proof_ids
            or (
                receipt is not None
                and set(plan.required_proof_ids).issubset(set(receipt.proof_receipt_ids))
            )
        ),
        "rollback_plan_bound": bool(plan.rollback_plan_id),
        "repair_succeeded": receipt is not None
        and receipt.terminal_status is TerminalStatus.SUCCEEDED,
        "validation_evidence_current": receipt is not None and bool(receipt.validation_receipt_ids),
        "changed_paths_within_predicted": receipt is not None
        and bool(receipt.changed_paths)
        and all(path in set(plan.predicted_files) for path in receipt.changed_paths),
        "envelope_identity_bound": plan.patch_envelope_id == envelope.envelope_id
        and (receipt is None or receipt.envelope_id == envelope.envelope_id),
        "adversarial_assurance_current": receipt is not None
        and bool(receipt.adversarial_assurance_receipt_ids),
        "receipt_does_not_authorize_merge": receipt is not None
        and receipt.authorizes_merge is False,
    }
    if set(checks) != set(LOW_RISK_MERGE_CONDITIONS):
        raise RepairControllerError("low-risk merge condition vocabulary is incomplete")
    satisfied = tuple(name for name in LOW_RISK_MERGE_CONDITIONS if checks[name])
    missing = tuple(name for name in LOW_RISK_MERGE_CONDITIONS if not checks[name])
    return satisfied, missing


def merge_disposition_for(
    *,
    policy: AutonomyPolicy,
    envelope: AutonomyEnvelope,
    plan: AutonomousRepairPlan,
    receipt: AutonomousRepairReceipt | None,
    isolated_worktree: bool,
) -> tuple[MergeDisposition, tuple[str, ...], tuple[str, ...]]:
    """R2 merge is a conjunction; R3 and above remain proposal-only."""

    risk = plan.risk_class
    if risk.rank >= RiskClass.R3_BOUNDED_REPOSITORY_MUTATION.rank:
        return MergeDisposition.PROPOSAL_ONLY, (), LOW_RISK_MERGE_CONDITIONS
    satisfied, missing = evaluate_low_risk_merge_conditions(
        policy=policy,
        envelope=envelope,
        plan=plan,
        receipt=receipt,
        isolated_worktree=isolated_worktree,
    )
    if risk is RiskClass.R2_REVERSIBLE_LOCAL and not missing:
        return MergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE, satisfied, missing
    if missing:
        return MergeDisposition.NOT_ELIGIBLE, satisfied, missing
    return MergeDisposition.NOT_ELIGIBLE, satisfied, missing


class AutonomousRepairController:
    """Tier-selection and scope facade over ``AutonomousRepairEngine@1``.

    The controller never implements IR, doctor, surface resolution, or byte
    mutation.  Those remain on the existing engine and materializer.  Effect
    admission remains ``DecisionRuntime``.
    """

    INTERFACE: Final[str] = AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    def __init__(
        self,
        *,
        engine: AutonomousRepairEngine,
        policy: AutonomyPolicy | None = None,
        decision_runtime: DecisionRuntime | None = None,
        materializer: AutonomousRepairMaterializer | None = None,
        model_caller: Callable[[RepairRequest], Mapping[str, Any]] | None = None,
        max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES,
        base_backoff_milliseconds: int = DEFAULT_BASE_BACKOFF_MILLISECONDS,
    ) -> None:
        if not isinstance(engine, AutonomousRepairEngine):
            raise RepairControllerError("engine must be the existing AutonomousRepairEngine")
        if policy is not None and not isinstance(policy, AutonomyPolicy):
            raise RepairControllerError("policy must be an AutonomyPolicy or None")
        if decision_runtime is not None and not isinstance(decision_runtime, DecisionRuntime):
            raise RepairControllerError("decision_runtime must be DecisionRuntime or None")
        if materializer is not None and not isinstance(materializer, AutonomousRepairMaterializer):
            raise RepairControllerError(
                "materializer must be AutonomousRepairMaterializer or None"
            )
        if model_caller is not None and not callable(model_caller):
            raise RepairControllerError("model_caller must be callable or None")
        self._engine = engine
        self._policy = policy
        self._decision_runtime = decision_runtime
        self._materializer = materializer
        self._model_caller = model_caller
        self._max_identical_failures = _int(
            max_identical_failures, "max_identical_failures", minimum=1, maximum=32
        )
        self._base_backoff_milliseconds = _int(
            base_backoff_milliseconds, "base_backoff_milliseconds", minimum=0
        )
        self._failures: dict[str, FailureMemoryRecord] = {}
        self._model_calls = 0

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine(self) -> AutonomousRepairEngine:
        return self._engine

    @property
    def engine_interface(self) -> str:
        return AUTONOMOUS_REPAIR_INTERFACE

    @property
    def decision_runtime(self) -> DecisionRuntime | None:
        return self._decision_runtime

    @property
    def materializer(self) -> AutonomousRepairMaterializer | None:
        return self._materializer

    @property
    def model_call_count(self) -> int:
        return self._model_calls

    @property
    def protected_authority_paths(self) -> tuple[str, ...]:
        return PROTECTED_AUTHORITY_PATHS

    def select_tier(self, request: RepairRequest) -> RepairTier:
        """Select the requested closed tier after its preconditions hold."""

        if not isinstance(request, RepairRequest):
            raise RepairControllerError("request must be a RepairRequest")
        requested = request.requested_tier
        if requested is RepairTier.MODEL_ASSISTED_BOUNDED:
            codes = self._model_assisted_gaps(request)
            if codes:
                raise RepairControllerError(
                    "model-assisted repair is missing required bindings: " + ", ".join(codes)
                )
        return requested

    def admit(self, request: RepairRequest) -> RepairControllerResult:
        """Fail closed on scope escape, self-edit, and authority-path mutation."""

        return self._evaluate(request, execute=False)

    def admit_source_edit(self, request: RepairRequest) -> RepairControllerResult:
        """Admit or reject one exact source-edit operator without writing bytes."""

        if request.source_edit_operator is None:
            return self._result(
                request,
                RepairControllerDisposition.REJECTED_SOURCE_EDIT,
                reason_codes=("source_edit_operator_missing",),
            )
        return self._evaluate(request, execute=False, source_edit_only=True)

    def repair(self, request: RepairRequest) -> RepairControllerResult:
        """Admit, select a tier, and delegate to the existing repair engine."""

        return self._evaluate(request, execute=request.execute)

    def observe_failure(self, request: RepairRequest, *, diagnostic_receipt_id: str = "") -> None:
        """Record one identical-failure observation without calling a model."""

        if not isinstance(request, RepairRequest):
            raise RepairControllerError("request must be a RepairRequest")
        signature = request.failure_signature
        if not signature:
            raise RepairControllerError("failure_signature is required to observe a failure")
        diagnostic = diagnostic_receipt_id or request.diagnostic_receipt_id or signature
        self._record_failure(request, diagnostic_receipt_id=diagnostic)

    def _evaluate(
        self,
        request: RepairRequest,
        *,
        execute: bool,
        source_edit_only: bool = False,
    ) -> RepairControllerResult:
        if not isinstance(request, RepairRequest):
            raise RepairControllerError("request must be a RepairRequest")
        if self._policy is not None and request.policy.policy_id != self._policy.policy_id:
            raise RepairControllerError("request policy does not match the bound controller policy")

        suffix_block = self._suffix_block(request)
        if suffix_block is not None:
            return suffix_block

        path_codes = self._path_rejection_codes(request)
        if "self_edit" in path_codes:
            return self._result(
                request,
                RepairControllerDisposition.REJECTED_SELF_EDIT,
                reason_codes=("self_edit", *path_codes),
            )
        if "validator_policy_key" in path_codes or "protected_authority_path" in path_codes:
            return self._result(
                request,
                RepairControllerDisposition.REJECTED_VALIDATOR_POLICY_KEY,
                reason_codes=("validator_policy_key", *path_codes),
            )
        if "scope_escape" in path_codes:
            return self._result(
                request,
                RepairControllerDisposition.REJECTED_SCOPE_ESCAPE,
                reason_codes=("scope_escape", *path_codes),
            )

        symbol_codes = self._symbol_rejection_codes(request)
        if symbol_codes:
            return self._result(
                request,
                RepairControllerDisposition.REJECTED_SCOPE_ESCAPE,
                reason_codes=symbol_codes,
            )

        if request.requested_tier is RepairTier.MODEL_ASSISTED_BOUNDED:
            gaps = self._model_assisted_gaps(request)
            if "isolated_worktree_required" in gaps:
                return self._result(
                    request,
                    RepairControllerDisposition.REJECTED_MISSING_WORKTREE,
                    reason_codes=gaps,
                )
            if "sufficient_context_required" in gaps:
                return self._result(
                    request,
                    RepairControllerDisposition.REJECTED_INSUFFICIENT_CONTEXT,
                    reason_codes=gaps,
                )
            if gaps:
                return self._result(
                    request,
                    RepairControllerDisposition.REJECTED_MISSING_CHECKS,
                    reason_codes=gaps,
                )

        source_edit_admitted = False
        operator: AdmittedSourceEditOperator | None = None
        if request.source_edit_operator is not None:
            try:
                operator = AdmittedSourceEditOperator.from_mapping(request.source_edit_operator)
            except AdmittedSourceEditError as exc:
                return self._result(
                    request,
                    RepairControllerDisposition.REJECTED_SOURCE_EDIT,
                    reason_codes=(str(exc) or "source_edit_operator_not_admitted",),
                )
            operator_code = classify_forbidden_path(
                operator.relative_path,
                allowed_paths=request.allowed_paths or request.envelope.allowed_paths,
                extra_protected=request.extra_protected_paths,
            )
            if operator_code == "self_edit":
                return self._result(
                    request,
                    RepairControllerDisposition.REJECTED_SELF_EDIT,
                    reason_codes=("self_edit", "source_edit_self_edit"),
                )
            if operator_code in {"validator_policy_key", "protected_authority_path"}:
                return self._result(
                    request,
                    RepairControllerDisposition.REJECTED_VALIDATOR_POLICY_KEY,
                    reason_codes=("validator_policy_key", "source_edit_authority_path"),
                )
            if operator_code == "scope_escape":
                return self._result(
                    request,
                    RepairControllerDisposition.REJECTED_SCOPE_ESCAPE,
                    reason_codes=("scope_escape", "source_edit_scope_escape"),
                )
            source_edit_admitted = True
            if source_edit_only:
                plan = self._plan_for(request)
                return self._result(
                    request,
                    RepairControllerDisposition.ADMITTED,
                    plan=plan,
                    reason_codes=("source_edit_admitted",),
                    source_edit_admitted=True,
                )

        identical = self._identical_failure_result(request)
        if identical is not None:
            return identical

        model_calls_before = self._model_calls
        if (
            request.requested_tier is RepairTier.MODEL_ASSISTED_BOUNDED
            and self._model_caller is not None
        ):
            payload = self._model_caller(request)
            if not isinstance(payload, Mapping):
                raise RepairControllerError("model_caller must return a mapping")
            _reject_forbidden_keys(payload, "model_caller")
            self._model_calls += 1

        plan = self._plan_for(request)
        engine_report: AutonomousRepairReport | None = None
        materializer_invoked = False
        changed_paths = request.changed_paths or plan.predicted_files
        if execute:
            engine_report = self._engine.run(self._work_items_for(request, plan))
            if operator is not None:
                if self._decision_runtime is None:
                    return self._result(
                        request,
                        RepairControllerDisposition.REJECTED_SOURCE_EDIT,
                        plan=plan,
                        reason_codes=("decision_runtime_required",),
                        source_edit_admitted=source_edit_admitted,
                        model_call_count=self._model_calls - model_calls_before,
                    )
                if self._materializer is None:
                    return self._result(
                        request,
                        RepairControllerDisposition.REJECTED_SOURCE_EDIT,
                        plan=plan,
                        reason_codes=("materializer_required",),
                        source_edit_admitted=source_edit_admitted,
                        model_call_count=self._model_calls - model_calls_before,
                    )
                self._materializer.materialize_plans(
                    [
                        {
                            "plan_id": plan.plan_id,
                            "work_id": request.envelope.task_id,
                            "operation": request.predicted_symbols[0],
                            "materialize_ready": True,
                            "preferred_path": operator.relative_path,
                            "handler": request.predicted_symbols[0],
                            "source_edit_operator": dict(request.source_edit_operator or {}),
                        }
                    ]
                )
                materializer_invoked = True
                changed_paths = (operator.relative_path,)

        terminal = TerminalStatus.SUCCEEDED
        validation_ids = request.validation_receipt_ids
        if execute and engine_report is not None and engine_report.llm_used:
            terminal = TerminalStatus.BLOCKED
            validation_ids = ()
        if execute and not validation_ids:
            terminal = TerminalStatus.PENDING
        receipt = None
        if terminal is not TerminalStatus.SUCCEEDED or validation_ids:
            receipt = AutonomousRepairReceipt(
                plan_id=plan.plan_id,
                envelope_id=request.envelope.envelope_id,
                terminal_status=terminal if validation_ids else TerminalStatus.PENDING,
                changed_paths=changed_paths if terminal is TerminalStatus.SUCCEEDED else (),
                validation_receipt_ids=validation_ids,
                proof_receipt_ids=request.proof_receipt_ids,
                adversarial_assurance_receipt_ids=request.adversarial_assurance_receipt_ids,
                rollback_receipt_id=plan.rollback_plan_id if terminal is TerminalStatus.FAILED else "",
                failure_signature=request.failure_signature,
                diagnostic_receipt_id=request.diagnostic_receipt_id,
                authorizes_merge=False,
            )
            if receipt.terminal_status is not TerminalStatus.SUCCEEDED:
                receipt = AutonomousRepairReceipt(
                    plan_id=plan.plan_id,
                    envelope_id=request.envelope.envelope_id,
                    terminal_status=receipt.terminal_status,
                    changed_paths=(),
                    validation_receipt_ids=receipt.validation_receipt_ids,
                    proof_receipt_ids=receipt.proof_receipt_ids,
                    adversarial_assurance_receipt_ids=receipt.adversarial_assurance_receipt_ids,
                    failure_signature=request.failure_signature,
                    diagnostic_receipt_id=request.diagnostic_receipt_id,
                    authorizes_merge=False,
                )

        if request.failure_signature and (
            receipt is None or receipt.terminal_status is not TerminalStatus.SUCCEEDED
        ):
            self._record_failure(request, diagnostic_receipt_id=request.diagnostic_receipt_id)
        elif request.failure_signature and receipt is not None:
            self._failures.pop(request.failure_key(), None)

        merge, satisfied, missing = merge_disposition_for(
            policy=request.policy,
            envelope=request.envelope,
            plan=plan,
            receipt=receipt,
            isolated_worktree=request.isolated_worktree,
        )
        rolled_back = False
        if merge is MergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE:
            disposition = RepairControllerDisposition.MERGE_ELIGIBLE
        elif merge is MergeDisposition.PROPOSAL_ONLY:
            disposition = RepairControllerDisposition.PROPOSAL_ONLY
        elif execute:
            disposition = RepairControllerDisposition.ENGINE_DELEGATED
        else:
            disposition = RepairControllerDisposition.ADMITTED
        if materializer_invoked and (
            receipt is None or receipt.terminal_status is not TerminalStatus.SUCCEEDED
        ):
            disposition = RepairControllerDisposition.ROLLED_BACK
            rolled_back = True
            merge = MergeDisposition.REJECTED
        reason = [
            "tier:" + plan.repair_tier.value,
            "engine:" + AUTONOMOUS_REPAIR_INTERFACE,
        ]
        if source_edit_admitted:
            reason.append("source_edit_admitted")
        if execute:
            reason.append("engine_delegated")
        if rolled_back:
            reason.append("rolled_back")
        return self._result(
            request,
            disposition,
            plan=plan,
            receipt=receipt,
            merge_disposition=merge,
            satisfied_merge_conditions=satisfied,
            missing_merge_conditions=missing,
            reason_codes=tuple(reason),
            model_call_count=self._model_calls - model_calls_before,
            engine_report=None if engine_report is None else engine_report.to_dict(),
            source_edit_admitted=source_edit_admitted,
            materializer_invoked=materializer_invoked,
            rolled_back=rolled_back,
        )

    def _path_rejection_codes(self, request: RepairRequest) -> tuple[str, ...]:
        allowed = request.allowed_paths or request.envelope.allowed_paths
        codes: list[str] = []
        seen: set[str] = set()
        for path in request.candidate_paths:
            code = classify_forbidden_path(
                path,
                allowed_paths=allowed,
                extra_protected=request.extra_protected_paths,
            )
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
        if request.plan is not None:
            for path in request.plan.predicted_files:
                code = classify_forbidden_path(
                    path,
                    allowed_paths=allowed,
                    extra_protected=request.extra_protected_paths,
                )
                if code and code not in seen:
                    seen.add(code)
                    codes.append(code)
        return tuple(codes)

    def _symbol_rejection_codes(self, request: RepairRequest) -> tuple[str, ...]:
        forbidden = set(request.forbidden_symbols)
        if request.plan is not None:
            forbidden.update(request.plan.forbidden_symbols)
        hits = [symbol for symbol in request.predicted_symbols if symbol in forbidden]
        if hits:
            return ("forbidden_symbol", *tuple(sorted(hits)))
        allowed_symbols = request.envelope.allowed_symbols
        if allowed_symbols and not all(
            symbol in set(allowed_symbols) for symbol in request.predicted_symbols
        ):
            return ("symbol_scope_escape",)
        return ()

    def _model_assisted_gaps(self, request: RepairRequest) -> tuple[str, ...]:
        gaps: list[str] = []
        if not request.predicted_files:
            gaps.append("exact_files_required")
        if not request.predicted_symbols:
            gaps.append("exact_symbols_required")
        if not (request.isolated_worktree and request.effective_worktree_id):
            gaps.append("isolated_worktree_required")
        if not (request.context_sufficient and request.context_reference_ids):
            gaps.append("sufficient_context_required")
        if not request.required_test_ids:
            gaps.append("predetermined_tests_required")
        if not request.effective_rollback_plan_id:
            gaps.append("rollback_plan_required")
        if not request.envelope.envelope_id:
            gaps.append("patch_envelope_required")
        return tuple(gaps)

    def _suffix_block(self, request: RepairRequest) -> RepairControllerResult | None:
        suffix = request.suffix_receipt
        if suffix is None:
            return None
        if suffix.disposition in _SUFFIX_BACKOFF_DISPOSITIONS:
            disposition = (
                RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
                if suffix.disposition is RecedingHorizonDisposition.IDENTICAL_FAILURE_EXHAUSTED
                else RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
            )
            return self._result(
                request,
                disposition,
                reason_codes=("suffix_identical_failure", suffix.disposition.value),
                diagnostic_reused=suffix.diagnostic_reused or True,
                backoff_milliseconds=suffix.backoff_milliseconds,
            )
        if (
            suffix.disposition is RecedingHorizonDisposition.PREFIX_PRESERVED
            and not suffix.invalidated_step_ids
        ):
            return self._result(
                request,
                RepairControllerDisposition.REJECTED_MISSING_ENVELOPE,
                reason_codes=("no_suffix_to_repair",),
            )
        return None

    def _identical_failure_result(self, request: RepairRequest) -> RepairControllerResult | None:
        if not request.failure_signature:
            return None
        record = self._failures.get(request.failure_key())
        if record is None:
            return None
        plan = self._plan_for(request)
        if record.attempts >= self._max_identical_failures:
            return self._result(
                request,
                RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED,
                plan=plan,
                reason_codes=("identical_failure_exhausted", record.failure_signature),
                diagnostic_reused=True,
                backoff_milliseconds=record.backoff_milliseconds,
            )
        backoff = self._base_backoff_milliseconds * (2 ** record.attempts)
        updated = FailureMemoryRecord(
            failure_signature=record.failure_signature,
            diagnostic_receipt_id=record.diagnostic_receipt_id,
            attempts=record.attempts + 1,
            backoff_milliseconds=backoff,
            last_tier=request.requested_tier,
        )
        self._failures[request.failure_key()] = updated
        return self._result(
            request,
            RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF,
            plan=plan,
            reason_codes=("identical_failure_backoff", record.diagnostic_receipt_id),
            diagnostic_reused=True,
            backoff_milliseconds=backoff,
        )

    def _record_failure(self, request: RepairRequest, *, diagnostic_receipt_id: str) -> None:
        key = request.failure_key()
        existing = self._failures.get(key)
        attempts = 1 if existing is None else existing.attempts + 1
        backoff = self._base_backoff_milliseconds * (2 ** max(attempts - 1, 0))
        diagnostic = diagnostic_receipt_id or (existing.diagnostic_receipt_id if existing else "")
        if not diagnostic:
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
            diagnostic = f"diagnostic:{digest}"
        self._failures[key] = FailureMemoryRecord(
            failure_signature=request.failure_signature,
            diagnostic_receipt_id=diagnostic,
            attempts=attempts,
            backoff_milliseconds=backoff,
            last_tier=request.requested_tier,
        )

    def _plan_for(self, request: RepairRequest) -> AutonomousRepairPlan:
        if request.plan is not None:
            if request.plan.patch_envelope_id != request.envelope.envelope_id:
                raise RepairControllerError("repair plan envelope identity does not match")
            if request.plan.objective_id != request.envelope.objective_id:
                raise RepairControllerError("repair plan objective identity does not match")
            return request.plan
        worktree = request.worktree_id or "worktree:unbound"
        rollback = request.rollback_plan_id or "rollback:unbound"
        context_ids = request.context_reference_ids or ("context:unbound",)
        return AutonomousRepairPlan(
            objective_id=request.envelope.objective_id,
            task_id=request.envelope.task_id,
            repair_tier=request.requested_tier,
            predicted_files=request.predicted_files,
            predicted_symbols=request.predicted_symbols,
            patch_envelope_id=request.envelope.envelope_id,
            context_reference_ids=context_ids,
            required_test_ids=request.required_test_ids or request.envelope.required_test_ids,
            required_proof_ids=request.required_proof_ids or request.envelope.required_proof_ids,
            worktree_id=worktree,
            allowed_paths=request.envelope.allowed_paths,
            forbidden_symbols=request.forbidden_symbols,
            rollback_plan_id=rollback,
            risk_class=request.envelope.risk_assessment.risk_class,
            max_changed_files=request.max_changed_files,
            max_changed_lines=request.max_changed_lines,
        )

    def _work_items_for(
        self, request: RepairRequest, plan: AutonomousRepairPlan
    ) -> list[RepairWorkItem]:
        if request.work_items:
            return [
                item if isinstance(item, RepairWorkItem) else RepairWorkItem.from_mapping(item)
                for item in request.work_items
            ]
        items: list[RepairWorkItem] = []
        for path, symbol in zip(plan.predicted_files, plan.predicted_symbols):
            items.append(
                RepairWorkItem(
                    work_id=f"work:{request.envelope.task_id}:{symbol}",
                    operation=symbol,
                    kind="bounded_repair",
                    contract_id=f"surface:{symbol}",
                    path=path,
                    symbol=symbol,
                    write_paths=(path,),
                    reason_codes=("bounded_autonomous_repair",),
                    domain="agent_supervisor",
                )
            )
        if not items:
            items.append(
                RepairWorkItem(
                    work_id=f"work:{request.envelope.task_id}",
                    operation=plan.predicted_symbols[0],
                    path=plan.predicted_files[0],
                    symbol=plan.predicted_symbols[0],
                    write_paths=plan.predicted_files,
                    reason_codes=("bounded_autonomous_repair",),
                )
            )
        return items

    def _result(
        self,
        request: RepairRequest,
        disposition: RepairControllerDisposition,
        *,
        plan: AutonomousRepairPlan | None = None,
        receipt: AutonomousRepairReceipt | None = None,
        merge_disposition: MergeDisposition = MergeDisposition.NOT_ELIGIBLE,
        satisfied_merge_conditions: Sequence[str] = (),
        missing_merge_conditions: Sequence[str] = (),
        reason_codes: Sequence[str] = (),
        diagnostic_reused: bool = False,
        backoff_milliseconds: int = 0,
        model_call_count: int = 0,
        engine_report: Mapping[str, Any] | None = None,
        source_edit_admitted: bool = False,
        materializer_invoked: bool = False,
        rolled_back: bool = False,
    ) -> RepairControllerResult:
        selected = request.requested_tier if plan is None else plan.repair_tier
        return RepairControllerResult(
            disposition=disposition,
            selected_tier=selected,
            reason_codes=reason_codes,
            plan=plan,
            receipt=receipt,
            merge_disposition=merge_disposition,
            satisfied_merge_conditions=tuple(satisfied_merge_conditions),
            missing_merge_conditions=tuple(missing_merge_conditions),
            diagnostic_reused=diagnostic_reused,
            backoff_milliseconds=backoff_milliseconds,
            model_call_count=model_call_count,
            engine_report=engine_report,
            source_edit_admitted=source_edit_admitted,
            materializer_invoked=materializer_invoked,
            rolled_back=rolled_back,
        )


__all__ = [
    "AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE",
    "AUTONOMOUS_REPAIR_CONTROLLER_SCHEMA",
    "LOW_RISK_MERGE_CONDITIONS",
    "PROTECTED_AUTHORITY_PATHS",
    "REPAIR_CONTROLLER_MODULE_REL",
    "SELF_EDIT_PATHS",
    "AutonomousRepairController",
    "FailureMemoryRecord",
    "MergeDisposition",
    "RepairControllerDisposition",
    "RepairControllerError",
    "RepairControllerResult",
    "RepairRequest",
    "classify_forbidden_path",
    "evaluate_low_risk_merge_conditions",
    "is_protected_authority_path",
    "is_self_edit_path",
    "is_validator_policy_key_path",
    "merge_disposition_for",
    "path_under_any",
    "path_under_prefix",
]
