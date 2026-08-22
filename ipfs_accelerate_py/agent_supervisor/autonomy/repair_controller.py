# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded facade over the existing autonomous-repair engine.

``AutonomousRepairController@1`` selects among deterministic, template-
constrained, and model-assisted tiers and binds an exact envelope, isolated
worktree, predetermined checks, identical-failure backoff, and merge
disposition.  It is not a second repair engine, materializer, worktree
manager, DecisionRuntime, or merge authority.

Admission and execution stay with ``AutonomousRepairEngine``,
``AdmittedSourceEditOperator`` / ``AutonomousRepairMaterializer``, and
``DecisionRuntime``.  Repair receipts are evidence: they never authorize
merge or an effect.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..autonomous_repair.contracts import (
    AutonomousRepairPolicy,
    AutonomousRepairReport,
    RepairWorkItem,
)
from ..autonomous_repair.engine import AutonomousRepairEngine
from ..autonomous_repair.materialize import (
    AdmittedSourceEditError,
    AdmittedSourceEditOperator,
    AutonomousRepairMaterializer,
    MaterializePolicy,
)
from ..proof.formal_verification_contracts import canonical_json, content_identity
from .contracts import (
    AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
    MAX_CANONICAL_RECORD_BYTES,
    MAX_IDENTIFIER_BYTES,
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
REPAIR_CONTROLLER_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-request@1"
)
REPAIR_CONTROLLER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-result@1"
)
REPAIR_FAILURE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-failure-record@1"
)
MAX_REPAIR_CONTROLLER_RECORD_BYTES: Final[int] = 4 * MAX_CANONICAL_RECORD_BYTES
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 3
DEFAULT_BASE_BACKOFF_MS: Final[int] = 10
DEFAULT_MAX_BACKOFF_MS: Final[int] = 40
ENGINE_AUTHORITY: Final[str] = "ipfs_accelerate_py.agent_supervisor.autonomous_repair"

SELF_PROTECTING_PATHS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",
    "ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair",
    "ipfs_accelerate_py/agent_supervisor/context/decision_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/control/execution_permit.py",
)
SIBLING_REPOSITORY_PREFIXES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py",
    "ipfs_kit_py",
    "ipfs_model_manager_py",
    "ipfs_transformers_py",
    "external",
)
VALIDATOR_POLICY_KEY_SEGMENTS: Final[frozenset[str]] = frozenset(
    {
        "verification",
        "verifier",
        "verifiers",
        "policy",
        "policies",
        "oracle",
        "oracles",
        "trusted_keys",
        "secrets",
        "credentials",
        "private_keys",
        "signing_keys",
        "validator",
        "validators",
    }
)
VALIDATOR_POLICY_KEY_SUFFIXES: Final[tuple[str, ...]] = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
)
VALIDATOR_POLICY_KEY_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        "authorized_keys",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "policy.json",
        "policy.yaml",
        "policy.yml",
        "oracle.json",
        "benchmark_oracle.json",
        "golden.json",
    }
)
VALIDATOR_POLICY_KEY_PREFIXES: Final[tuple[str, ...]] = (
    ".ssh",
    ".aws",
    ".gnupg",
    "config",
    "secrets",
    "credentials",
    "ipfs_accelerate_py/agent_supervisor/verification",
    "ipfs_accelerate_py/agent_supervisor/proof",
    "ipfs_accelerate_py/agent_supervisor/validation",
)
LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "policy_autonomous_merge_enabled",
    "risk_at_most_r2",
    "reversible",
    "autonomy_level_allows_execute_reversible",
    "no_security_or_protocol_sensitivity",
    "no_irreversible_or_legal_effect",
    "envelope_paths_bound",
    "changed_paths_inside_envelope",
    "changed_files_within_bound",
    "changed_lines_within_bound",
    "required_tests_present",
    "required_proofs_present",
    "validation_receipts_present",
    "rollback_plan_present",
    "no_scope_escape",
    "no_self_edit",
    "no_validator_policy_key_mutation",
    "no_sibling_repository_mutation",
    "predetermined_checks_bound",
    "isolated_worktree_bound",
    "repair_succeeded",
    "no_pending_source_edit",
    "receipt_does_not_authorize_merge",
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

ModelInvoker = Callable[[AutonomousRepairPlan, AutonomyEnvelope], Mapping[str, Any]]


class RepairControllerError(ValueError):
    """Raised when repair-controller inputs themselves are malformed."""


class RepairControllerDisposition(str, Enum):
    """Closed outcome of one facade step.  None grants merge or effect."""

    ENGINE_DELEGATED = "engine_delegated"
    MODEL_ASSISTED_DELEGATED = "model_assisted_delegated"
    SOURCE_EDIT_VALIDATION_PENDING = "source_edit_validation_pending"
    REJECTED = "rejected"
    IDENTICAL_FAILURE_REUSED = "identical_failure_reused"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    ROLLBACK_REQUIRED = "rollback_required"


class MergeDisposition(str, Enum):
    """Typed merge advice.  The facade never merges."""

    R2_AUTONOMOUS_MERGE_ELIGIBLE = "r2_autonomous_merge_eligible"
    R3_PROPOSAL = "r3_proposal"
    NOT_ELIGIBLE = "not_eligible"
    HUMAN_REQUIRED = "human_required"
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


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RepairControllerError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise RepairControllerError(f"{name} must be an integer of at least {minimum}")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise RepairControllerError(f"{name} must be one of: {allowed}") from exc


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


def _posix_path(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    result = value.strip().replace("\\", "/")
    if not result or "\x00" in result:
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    parsed = PurePosixPath(result)
    if parsed.is_absolute() or ".." in parsed.parts or result in {".", ""}:
        raise RepairControllerError(f"{name} must be a repository-relative POSIX path")
    return parsed.as_posix()


def _paths(value: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    raw = _identifiers(value, name, required=required, preserve_order=True)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        path = _posix_path(item, name)
        if path not in seen:
            seen.add(path)
            normalized.append(path)
    return tuple(sorted(normalized))


def path_is_under(path: str, prefix: str) -> bool:
    """Return True when *path* equals *prefix* or is a descendant of it."""

    candidate = path.replace("\\", "/").strip("/")
    bound = prefix.replace("\\", "/").strip("/")
    if not bound:
        return False
    return candidate == bound or candidate.startswith(bound + "/")


def path_is_allowed(path: str, prefixes: Sequence[str]) -> bool:
    return any(path_is_under(path, prefix) for prefix in prefixes)


def path_is_self_edit(path: str) -> bool:
    return path_is_allowed(path, SELF_PROTECTING_PATHS)


def path_is_sibling_repository(path: str) -> bool:
    return path_is_allowed(path, SIBLING_REPOSITORY_PREFIXES)


def path_is_validator_policy_key(path: str) -> bool:
    lower = path.replace("\\", "/").strip().lower()
    if path_is_allowed(lower, VALIDATOR_POLICY_KEY_PREFIXES):
        return True
    parsed = PurePosixPath(lower)
    if parsed.name in VALIDATOR_POLICY_KEY_BASENAMES:
        return True
    if any(lower.endswith(suffix) for suffix in VALIDATOR_POLICY_KEY_SUFFIXES):
        return True
    return any(segment in VALIDATOR_POLICY_KEY_SEGMENTS for segment in parsed.parts)


def classify_forbidden_paths(paths: Sequence[str]) -> tuple[str, ...]:
    """Return closed reason codes for paths the facade must reject."""

    reasons: list[str] = []
    for path in paths:
        if path_is_self_edit(path) and "self_edit_rejected" not in reasons:
            reasons.append("self_edit_rejected")
        if (
            path_is_validator_policy_key(path)
            and "validator_policy_key_mutation_rejected" not in reasons
        ):
            reasons.append("validator_policy_key_mutation_rejected")
        if (
            path_is_sibling_repository(path)
            and "sibling_repository_mutation_rejected" not in reasons
        ):
            reasons.append("sibling_repository_mutation_rejected")
    return tuple(reasons)


def _scope_escape(paths: Sequence[str], allowed: Sequence[str]) -> bool:
    return any(not path_is_allowed(path, allowed) for path in paths)


def select_repair_tier(
    *,
    requested: RepairTier | None,
    deterministic_available: bool,
    template_available: bool,
    context_sufficient: bool,
    exact_files: bool,
    exact_symbols: bool,
    isolated_worktree: bool,
    predetermined_checks: bool,
) -> RepairTier:
    """Choose the cheapest admissible tier.  Never upgrades a requested tier."""

    model_ready = (
        exact_files
        and exact_symbols
        and isolated_worktree
        and predetermined_checks
        and context_sufficient
    )
    if requested is RepairTier.DETERMINISTIC:
        if not deterministic_available:
            raise RepairControllerError("deterministic repair is not available")
        return RepairTier.DETERMINISTIC
    if requested is RepairTier.TEMPLATE_CONSTRAINED:
        if not template_available:
            raise RepairControllerError("template-constrained repair is not available")
        return RepairTier.TEMPLATE_CONSTRAINED
    if requested is RepairTier.MODEL_ASSISTED_BOUNDED:
        if not model_ready:
            raise RepairControllerError(
                "model-assisted repair requires exact files/symbols, context, "
                "an isolated worktree, and predetermined checks"
            )
        return RepairTier.MODEL_ASSISTED_BOUNDED
    if requested is not None:
        raise RepairControllerError("unsupported repair tier")
    if deterministic_available:
        return RepairTier.DETERMINISTIC
    if template_available:
        return RepairTier.TEMPLATE_CONSTRAINED
    if model_ready:
        return RepairTier.MODEL_ASSISTED_BOUNDED
    raise RepairControllerError("no admissible repair tier")


def _backoff_milliseconds(count: int, *, base: int, maximum: int) -> int:
    if count <= 0:
        return 0
    delay = base
    for _ in range(count - 1):
        if delay >= maximum:
            return maximum
        delay *= 2
    return min(delay, maximum)


@dataclass(frozen=True)
class RepairFailureRecord:
    """Bounded identical-failure memory.  Never stores prompts or transcripts."""

    signature: str
    diagnostic_receipt_id: str
    count: int
    backoff_milliseconds: int
    last_seen_ms: int = 0

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
        object.__setattr__(self, "last_seen_ms", _int(self.last_seen_ms, "last_seen_ms"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_FAILURE_RECORD_SCHEMA,
            "signature": self.signature,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "count": self.count,
            "backoff_milliseconds": self.backoff_milliseconds,
            "last_seen_ms": self.last_seen_ms,
        }


def derive_failure_signature(
    *,
    envelope_id: str,
    predicted_files: Sequence[str],
    predicted_symbols: Sequence[str],
    failure_signature: str = "",
) -> str:
    if failure_signature:
        return _identifier(failure_signature, "failure_signature")
    return content_identity(
        {
            "envelope_id": envelope_id,
            "predicted_files": list(predicted_files),
            "predicted_symbols": list(predicted_symbols),
        }
    )


def evaluate_merge_conjunction(
    *,
    policy: AutonomyPolicy,
    envelope: AutonomyEnvelope,
    plan: AutonomousRepairPlan,
    changed_paths: Sequence[str],
    changed_line_count: int,
    validation_receipt_ids: Sequence[str],
    proof_receipt_ids: Sequence[str],
    terminal_status: TerminalStatus,
    source_edit_validation_pending: bool,
    forbidden_reasons: Sequence[str],
) -> Mapping[str, bool]:
    """Evaluate every stated low-risk merge condition.  Conjunction is exact."""

    risk = envelope.risk_assessment
    allowed = tuple(dict.fromkeys((*envelope.allowed_paths, *plan.allowed_paths)))
    changed = tuple(changed_paths)
    tests_ok = (not plan.required_test_ids) or bool(validation_receipt_ids)
    proofs_ok = (not plan.required_proof_ids) or set(plan.required_proof_ids).issubset(
        proof_receipt_ids
    )
    envelope_tests_ok = set(envelope.required_test_ids) == set(plan.required_test_ids)
    envelope_proofs_ok = set(envelope.required_proof_ids) == set(plan.required_proof_ids)
    succeeded = terminal_status is TerminalStatus.SUCCEEDED
    results = {
        "policy_autonomous_merge_enabled": policy.autonomous_merge_enabled is True,
        "risk_at_most_r2": risk.risk_class.rank <= RiskClass.R2_REVERSIBLE_LOCAL.rank,
        "reversible": envelope.reversible is True and risk.reversible is True,
        "autonomy_level_allows_execute_reversible": (
            envelope.autonomy_level.rank >= AutonomyLevel.EXECUTE_REVERSIBLE.rank
            and policy.allows(AutonomyLevel.EXECUTE_REVERSIBLE, risk.risk_class)
        ),
        "no_security_or_protocol_sensitivity": (
            not risk.security_sensitive and not risk.protocol_sensitive
        ),
        "no_irreversible_or_legal_effect": (
            not risk.irreversible_external_effect and not risk.legal_or_financial_effect
        ),
        "envelope_paths_bound": bool(envelope.allowed_paths) and bool(plan.allowed_paths),
        "changed_paths_inside_envelope": bool(changed)
        and not _scope_escape(changed, allowed),
        "changed_files_within_bound": 0 < len(changed) <= plan.max_changed_files,
        "changed_lines_within_bound": 0 < changed_line_count <= plan.max_changed_lines,
        "required_tests_present": tests_ok and bool(validation_receipt_ids),
        "required_proofs_present": proofs_ok,
        "validation_receipts_present": bool(validation_receipt_ids),
        "rollback_plan_present": bool(plan.rollback_plan_id),
        "no_scope_escape": "scope_escape_rejected" not in forbidden_reasons
        and not _scope_escape(changed, allowed)
        and not _scope_escape(plan.predicted_files, allowed),
        "no_self_edit": "self_edit_rejected" not in forbidden_reasons
        and not any(path_is_self_edit(path) for path in changed),
        "no_validator_policy_key_mutation": (
            "validator_policy_key_mutation_rejected" not in forbidden_reasons
            and not any(path_is_validator_policy_key(path) for path in changed)
        ),
        "no_sibling_repository_mutation": (
            "sibling_repository_mutation_rejected" not in forbidden_reasons
            and not any(path_is_sibling_repository(path) for path in changed)
        ),
        "predetermined_checks_bound": envelope_tests_ok and envelope_proofs_ok,
        "isolated_worktree_bound": bool(plan.worktree_id),
        "repair_succeeded": succeeded,
        "no_pending_source_edit": not source_edit_validation_pending,
        "receipt_does_not_authorize_merge": True,
    }
    if set(results) != set(LOW_RISK_MERGE_CONDITIONS):
        raise RepairControllerError("merge conjunction omitted a stated low-risk condition")
    return MappingProxyType({name: results[name] for name in LOW_RISK_MERGE_CONDITIONS})


def merge_disposition_for(
    conditions: Mapping[str, bool],
    *,
    risk_class: RiskClass,
    forbidden_reasons: Sequence[str],
) -> MergeDisposition:
    if forbidden_reasons:
        return MergeDisposition.REJECTED
    if all(conditions.values()):
        return MergeDisposition.R2_AUTONOMOUS_MERGE_ELIGIBLE
    if risk_class is RiskClass.R3_BOUNDED_REPOSITORY_MUTATION:
        return MergeDisposition.R3_PROPOSAL
    if risk_class.rank >= RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE.rank:
        return MergeDisposition.HUMAN_REQUIRED
    return MergeDisposition.NOT_ELIGIBLE


@dataclass(frozen=True)
class RepairControllerRequest:
    """One facade invocation.  Never a prompt, transcript, or raw source body."""

    envelope: AutonomyEnvelope
    policy: AutonomyPolicy
    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    worktree_id: str
    rollback_plan_id: str
    patch_envelope_id: str = ""
    context_reference_ids: tuple[str, ...] = ()
    required_test_ids: tuple[str, ...] = ()
    required_proof_ids: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    forbidden_symbols: tuple[str, ...] = ()
    max_changed_files: int = 1
    max_changed_lines: int = 100
    changed_paths: tuple[str, ...] = ()
    changed_line_count: int = 0
    requested_tier: RepairTier | None = None
    context_sufficient: bool = True
    template_available: bool = False
    deterministic_available: bool = True
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""
    source_edit_operator: Mapping[str, Any] | None = None
    work_items: tuple[Any, ...] = ()
    operation: str = ""
    now_ms: int = 0
    failed: bool = False
    apply_source_edit: bool = False
    delegate_to_engine: bool = True
    suffix_receipt_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, AutonomyEnvelope):
            raise RepairControllerError("envelope must be an AutonomyEnvelope")
        if not isinstance(self.policy, AutonomyPolicy):
            raise RepairControllerError("policy must be an AutonomyPolicy")
        object.__setattr__(
            self, "predicted_files", _paths(self.predicted_files, "predicted_files", required=True)
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
        patch_id = self.patch_envelope_id or self.envelope.envelope_id
        object.__setattr__(self, "patch_envelope_id", _identifier(patch_id, "patch_envelope_id"))
        object.__setattr__(
            self,
            "context_reference_ids",
            _identifiers(self.context_reference_ids, "context_reference_ids"),
        )
        tests = self.required_test_ids or self.envelope.required_test_ids
        proofs = self.required_proof_ids or self.envelope.required_proof_ids
        object.__setattr__(
            self, "required_test_ids", _identifiers(tests, "required_test_ids")
        )
        object.__setattr__(
            self, "required_proof_ids", _identifiers(proofs, "required_proof_ids")
        )
        allowed = self.allowed_paths or self.envelope.allowed_paths
        object.__setattr__(self, "allowed_paths", _paths(allowed, "allowed_paths", required=True))
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
        object.__setattr__(self, "changed_paths", _paths(self.changed_paths, "changed_paths"))
        object.__setattr__(
            self, "changed_line_count", _int(self.changed_line_count, "changed_line_count")
        )
        if self.requested_tier is not None:
            object.__setattr__(
                self, "requested_tier", _enum(self.requested_tier, RepairTier, "requested_tier")
            )
        for name in (
            "context_sufficient",
            "template_available",
            "deterministic_available",
            "failed",
            "apply_source_edit",
            "delegate_to_engine",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
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
        if self.source_edit_operator is not None and not isinstance(
            self.source_edit_operator, Mapping
        ):
            raise RepairControllerError("source_edit_operator must be an object")
        if self.source_edit_operator is not None:
            _reject_forbidden_keys(self.source_edit_operator, "source_edit_operator")
        if self.work_items is None:
            object.__setattr__(self, "work_items", ())
        elif isinstance(self.work_items, Sequence) and not isinstance(
            self.work_items, (str, bytes, bytearray)
        ):
            if len(self.work_items) > MAX_SEQUENCE_ITEMS:
                raise RepairControllerError("work_items contains too many items")
            object.__setattr__(self, "work_items", tuple(self.work_items))
        else:
            raise RepairControllerError("work_items must be a sequence")
        object.__setattr__(
            self, "operation", _identifier(self.operation, "operation", required=False)
        )
        object.__setattr__(self, "now_ms", _int(self.now_ms, "now_ms"))
        object.__setattr__(
            self,
            "suffix_receipt_id",
            _identifier(self.suffix_receipt_id, "suffix_receipt_id", required=False),
        )
        if self.policy.policy_id != self.envelope.policy_id:
            raise RepairControllerError("envelope policy identity does not match the supplied policy")
        if self.policy.authority_id != self.envelope.authority_id:
            raise RepairControllerError("envelope authority identity does not match the supplied policy")
        if not self.policy.allows(self.envelope.autonomy_level, self.envelope.risk_assessment.risk_class):
            raise RepairControllerError("envelope autonomy level exceeds the policy ceiling")

    @property
    def inspected_paths(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.predicted_files) | set(self.changed_paths)))


@dataclass(frozen=True)
class RepairControllerResult:
    """Facade result.  Evidence only; never a merge or effect permit."""

    disposition: RepairControllerDisposition
    selected_tier: RepairTier | None
    merge_disposition: MergeDisposition
    merge_conditions: Mapping[str, bool]
    reason_codes: tuple[str, ...]
    plan: AutonomousRepairPlan | None = None
    receipt: AutonomousRepairReceipt | None = None
    engine_report: Mapping[str, Any] | None = None
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    model_call_count: int = 0
    engine_run_count: int = 0
    source_edit_disposition: str = "not_source_edit"
    mutation_applied: bool = False
    validation_pending: bool = False
    engine_authority: str = ENGINE_AUTHORITY
    suffix_receipt_id: str = ""

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
            _enum(self.merge_disposition, MergeDisposition, "merge_disposition"),
        )
        if not isinstance(self.merge_conditions, Mapping):
            raise RepairControllerError("merge_conditions must be a mapping")
        if set(self.merge_conditions) != set(LOW_RISK_MERGE_CONDITIONS):
            raise RepairControllerError("merge_conditions must report every stated low-risk condition")
        object.__setattr__(
            self,
            "merge_conditions",
            MappingProxyType({key: bool(self.merge_conditions[key]) for key in LOW_RISK_MERGE_CONDITIONS}),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", preserve_order=True),
        )
        if self.plan is not None and not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan")
        if self.receipt is not None:
            if not isinstance(self.receipt, AutonomousRepairReceipt):
                raise RepairControllerError("receipt must be an AutonomousRepairReceipt")
            if self.receipt.authorizes_merge:
                raise RepairControllerError("repair receipts cannot independently authorize merge")
        if self.engine_report is not None:
            if not isinstance(self.engine_report, Mapping):
                raise RepairControllerError("engine_report must be an object")
            _reject_forbidden_keys(self.engine_report, "engine_report")
            object.__setattr__(self, "engine_report", MappingProxyType(dict(self.engine_report)))
        object.__setattr__(self, "diagnostic_reused", _bool(self.diagnostic_reused, "diagnostic_reused"))
        object.__setattr__(
            self, "backoff_milliseconds", _int(self.backoff_milliseconds, "backoff_milliseconds")
        )
        object.__setattr__(self, "model_call_count", _int(self.model_call_count, "model_call_count"))
        object.__setattr__(self, "engine_run_count", _int(self.engine_run_count, "engine_run_count"))
        object.__setattr__(
            self,
            "source_edit_disposition",
            _identifier(self.source_edit_disposition, "source_edit_disposition"),
        )
        object.__setattr__(self, "mutation_applied", _bool(self.mutation_applied, "mutation_applied"))
        object.__setattr__(
            self, "validation_pending", _bool(self.validation_pending, "validation_pending")
        )
        object.__setattr__(
            self, "engine_authority", _identifier(self.engine_authority, "engine_authority")
        )
        object.__setattr__(
            self,
            "suffix_receipt_id",
            _identifier(self.suffix_receipt_id, "suffix_receipt_id", required=False),
        )
        encoded = canonical_json(self.to_dict(include_identity=False)).encode("utf-8")
        if len(encoded) > MAX_REPAIR_CONTROLLER_RECORD_BYTES:
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
    def creates_repair_engine(self) -> bool:
        return False

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": REPAIR_CONTROLLER_RESULT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "disposition": self.disposition.value,
            "selected_tier": None if self.selected_tier is None else self.selected_tier.value,
            "merge_disposition": self.merge_disposition.value,
            "merge_conditions": dict(self.merge_conditions),
            "reason_codes": list(self.reason_codes),
            "plan": None if self.plan is None else self.plan.to_dict(),
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "engine_report": None if self.engine_report is None else dict(self.engine_report),
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "model_call_count": self.model_call_count,
            "engine_run_count": self.engine_run_count,
            "source_edit_disposition": self.source_edit_disposition,
            "mutation_applied": self.mutation_applied,
            "validation_pending": self.validation_pending,
            "engine_authority": self.engine_authority,
            "suffix_receipt_id": self.suffix_receipt_id,
            "authorizes_effect": False,
            "authorizes_merge": False,
            "creates_repair_engine": False,
        }
        if include_identity:
            payload["result_id"] = self.result_id
        return payload


def _empty_merge_conditions() -> Mapping[str, bool]:
    return MappingProxyType({name: False for name in LOW_RISK_MERGE_CONDITIONS})


class AutonomousRepairController:
    """Tier-selection and scope facade over ``AutonomousRepairEngine``.

    Effect execution remains ``DecisionRuntime``.  Merge remains the existing
    merge authorities.  Identical failures reuse diagnosis and never repeat a
    model call.
    """

    def __init__(
        self,
        *,
        engine: AutonomousRepairEngine | None = None,
        repo_root: str | Path | None = None,
        model_invoker: ModelInvoker | None = None,
        decision_runtime: Any = None,
        materializer: AutonomousRepairMaterializer | None = None,
        surface_files: Sequence[tuple[str, str | Path]] | None = None,
        max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES,
        base_backoff_milliseconds: int = DEFAULT_BASE_BACKOFF_MS,
        max_backoff_milliseconds: int = DEFAULT_MAX_BACKOFF_MS,
        engine_policy: AutonomousRepairPolicy | None = None,
    ) -> None:
        if engine is not None and not isinstance(engine, AutonomousRepairEngine):
            raise RepairControllerError("engine must be AutonomousRepairEngine")
        if materializer is not None and not isinstance(materializer, AutonomousRepairMaterializer):
            raise RepairControllerError("materializer must be AutonomousRepairMaterializer")
        if model_invoker is not None and not callable(model_invoker):
            raise RepairControllerError("model_invoker must be callable")
        self._engine = engine
        self._repo_root = None if repo_root is None else Path(repo_root)
        self._model_invoker = model_invoker
        self._decision_runtime = decision_runtime
        self._materializer = materializer
        self._surface_files = None if surface_files is None else tuple(surface_files)
        self._max_identical_failures = _int(
            max_identical_failures, "max_identical_failures", minimum=1
        )
        self._base_backoff_ms = _int(
            base_backoff_milliseconds, "base_backoff_milliseconds", minimum=1
        )
        self._max_backoff_ms = _int(
            max_backoff_milliseconds, "max_backoff_milliseconds", minimum=1
        )
        if self._base_backoff_ms > self._max_backoff_ms:
            raise RepairControllerError("base backoff cannot exceed max backoff")
        self._engine_policy = engine_policy or AutonomousRepairPolicy(
            apply_ir_logic=False,
            apply_doctor=False,
            require_zero_model_calls=True,
            allow_code_edit_materialize=False,
        )
        if self._engine_policy.allow_code_edit_materialize:
            raise RepairControllerError("policy flag is not source-edit admission")
        self._failures: dict[str, RepairFailureRecord] = {}
        self._model_call_count = 0
        self._engine_run_count = 0

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine(self) -> AutonomousRepairEngine:
        if self._engine is None:
            root = self._repo_root if self._repo_root is not None else Path(".")
            self._engine = AutonomousRepairEngine(
                repo_root=root,
                policy=self._engine_policy,
            )
        return self._engine

    @property
    def engine_type(self) -> type[AutonomousRepairEngine]:
        return AutonomousRepairEngine

    @property
    def decision_runtime(self) -> Any:
        return self._decision_runtime

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    @property
    def engine_run_count(self) -> int:
        return self._engine_run_count

    @property
    def failure_records(self) -> Mapping[str, RepairFailureRecord]:
        return MappingProxyType(dict(self._failures))

    def select_tier(self, request: RepairControllerRequest) -> RepairTier:
        return select_repair_tier(
            requested=request.requested_tier,
            deterministic_available=request.deterministic_available,
            template_available=request.template_available,
            context_sufficient=request.context_sufficient and bool(request.context_reference_ids),
            exact_files=bool(request.predicted_files),
            exact_symbols=bool(request.predicted_symbols),
            isolated_worktree=bool(request.worktree_id),
            predetermined_checks=bool(request.required_test_ids or request.required_proof_ids)
            or not request.envelope.required_test_ids,
        )

    def _forbidden_reasons(self, request: RepairControllerRequest) -> tuple[str, ...]:
        reasons = list(classify_forbidden_paths(request.inspected_paths))
        allowed = request.allowed_paths
        if _scope_escape(request.allowed_paths, request.envelope.allowed_paths):
            reasons.insert(0, "scope_escape_rejected")
        if _scope_escape(request.inspected_paths, request.envelope.allowed_paths):
            reasons.insert(0, "scope_escape_rejected")
        if _scope_escape(request.inspected_paths, allowed):
            reasons.insert(0, "scope_escape_rejected")
        predicted_symbols = set(request.predicted_symbols)
        if predicted_symbols.intersection(request.forbidden_symbols):
            reasons.append("forbidden_symbol_rejected")
        if not set(request.predicted_files).issubset(set(request.inspected_paths)):
            reasons.append("predicted_file_unbound")
        if request.apply_source_edit and request.source_edit_operator is None:
            reasons.append("source_edit_operator_missing")
        return tuple(dict.fromkeys(reasons))

    def _build_plan(
        self,
        request: RepairControllerRequest,
        tier: RepairTier,
    ) -> AutonomousRepairPlan:
        context_ids = request.context_reference_ids
        if not context_ids:
            if tier is RepairTier.MODEL_ASSISTED_BOUNDED:
                raise RepairControllerError("model-assisted repair requires sufficient context")
            context_ids = (f"context:{request.envelope.tree_id}",)
        try:
            return AutonomousRepairPlan(
                objective_id=request.envelope.objective_id,
                task_id=request.envelope.task_id,
                repair_tier=tier,
                predicted_files=request.predicted_files,
                predicted_symbols=request.predicted_symbols,
                patch_envelope_id=request.patch_envelope_id,
                context_reference_ids=context_ids,
                required_test_ids=request.required_test_ids,
                required_proof_ids=request.required_proof_ids,
                worktree_id=request.worktree_id,
                allowed_paths=request.allowed_paths,
                forbidden_symbols=request.forbidden_symbols,
                rollback_plan_id=request.rollback_plan_id,
                risk_class=request.envelope.risk_assessment.risk_class,
                max_changed_files=request.max_changed_files,
                max_changed_lines=request.max_changed_lines,
            )
        except AutonomyContractError as exc:
            raise RepairControllerError(str(exc)) from exc

    def _build_receipt(
        self,
        *,
        plan: AutonomousRepairPlan | None,
        request: RepairControllerRequest,
        status: TerminalStatus,
        changed_paths: Sequence[str] = (),
        failure_signature: str = "",
        diagnostic_receipt_id: str = "",
        validation_receipt_ids: Sequence[str] | None = None,
        rollback_receipt_id: str = "",
    ) -> AutonomousRepairReceipt | None:
        if plan is None:
            return None
        validation_ids = (
            request.validation_receipt_ids if validation_receipt_ids is None else validation_receipt_ids
        )
        try:
            return AutonomousRepairReceipt(
                plan_id=plan.plan_id,
                envelope_id=request.envelope.envelope_id,
                terminal_status=status,
                changed_paths=tuple(changed_paths),
                validation_receipt_ids=tuple(validation_ids),
                proof_receipt_ids=request.proof_receipt_ids,
                adversarial_assurance_receipt_ids=request.adversarial_assurance_receipt_ids,
                rollback_receipt_id=rollback_receipt_id,
                failure_signature=failure_signature,
                diagnostic_receipt_id=diagnostic_receipt_id,
                authorizes_merge=False,
            )
        except AutonomyContractError as exc:
            raise RepairControllerError(str(exc)) from exc

    def _work_items(self, request: RepairControllerRequest) -> tuple[RepairWorkItem, ...]:
        if request.work_items:
            return tuple(
                item if isinstance(item, RepairWorkItem) else RepairWorkItem.from_mapping(item)
                for item in request.work_items
            )
        operation = request.operation or request.envelope.task_id
        return (
            RepairWorkItem(
                work_id=request.envelope.task_id,
                operation=operation,
                path=request.predicted_files[0],
                symbol=request.predicted_symbols[0],
                write_paths=request.predicted_files,
                domain="agent_supervisor",
            ),
        )

    def _delegate_engine(self, request: RepairControllerRequest) -> AutonomousRepairReport | None:
        if not request.delegate_to_engine:
            return None
        report = self.engine.run(self._work_items(request))
        self._engine_run_count += 1
        if report.model_call_count:
            raise RepairControllerError("autonomous-repair engine must remain no-LLM")
        return report

    def _call_model(
        self,
        plan: AutonomousRepairPlan,
        request: RepairControllerRequest,
    ) -> Mapping[str, Any]:
        if self._model_invoker is None:
            return {}
        payload = self._model_invoker(plan, request.envelope)
        self._model_call_count += 1
        if not isinstance(payload, Mapping):
            raise RepairControllerError("model invoker must return an object")
        _reject_forbidden_keys(payload, "model_invoker")
        return payload

    def _admit_source_edit(
        self, request: RepairControllerRequest
    ) -> tuple[AdmittedSourceEditOperator | None, tuple[str, ...]]:
        if request.source_edit_operator is None:
            if request.apply_source_edit:
                return None, ("source_edit_operator_missing",)
            return None, ()
        try:
            operator = AdmittedSourceEditOperator.from_mapping(request.source_edit_operator)
        except AdmittedSourceEditError as exc:
            return None, (str(exc) or "source_edit_operator_not_admitted",)
        try:
            relative = _posix_path(operator.relative_path, "source_edit_operator.relative_path")
        except RepairControllerError:
            return operator, ("scope_escape_rejected", "source_edit_path_binding_mismatch")
        extra: list[str] = list(classify_forbidden_paths((relative,)))
        if _scope_escape((relative,), request.allowed_paths) or _scope_escape(
            (relative,), request.envelope.allowed_paths
        ):
            extra.insert(0, "scope_escape_rejected")
        if relative not in set(request.predicted_files):
            extra.append("source_edit_path_unbound")
        if extra:
            return operator, tuple(dict.fromkeys(extra))
        return operator, ()

    def _apply_source_edit(
        self,
        request: RepairControllerRequest,
        operator: AdmittedSourceEditOperator,
    ) -> Mapping[str, Any]:
        root = Path(self._repo_root if self._repo_root is not None else operator.owner_root)
        surface_files = self._surface_files
        if surface_files is None:
            target = root / operator.relative_path
            if target.is_file():
                surface_files = (("accelerate", target),)
        materializer = self._materializer or AutonomousRepairMaterializer(
            repo_root=root,
            surface_files=surface_files,
            policy=MaterializePolicy(write_data_catalog=False, dry_run=False),
        )
        plan = {
            "plan_id": request.patch_envelope_id,
            "work_id": request.envelope.task_id,
            "operation": request.operation or request.envelope.task_id,
            "materialize_ready": True,
            "preferred_path": operator.relative_path,
            "handler": request.predicted_symbols[0] if request.predicted_symbols else None,
            "source_edit_operator": dict(request.source_edit_operator or {}),
        }
        return materializer.materialize_plans([plan])

    def _record_failure(
        self,
        signature: str,
        *,
        diagnostic_receipt_id: str,
        now_ms: int,
    ) -> RepairFailureRecord:
        current = self._failures.get(signature)
        count = 1 if current is None else current.count + 1
        record = RepairFailureRecord(
            signature=signature,
            diagnostic_receipt_id=diagnostic_receipt_id or (
                current.diagnostic_receipt_id if current is not None else ""
            ),
            count=count,
            backoff_milliseconds=_backoff_milliseconds(
                count, base=self._base_backoff_ms, maximum=self._max_backoff_ms
            ),
            last_seen_ms=now_ms,
        )
        self._failures[signature] = record
        return record

    def run(self, request: RepairControllerRequest) -> RepairControllerResult:
        """Admit, bound, and dispatch one repair.  Never merges or grants effect."""

        if not isinstance(request, RepairControllerRequest):
            raise RepairControllerError("request must be a RepairControllerRequest")
        forbidden = list(self._forbidden_reasons(request))
        operator, source_reasons = self._admit_source_edit(request)
        forbidden.extend(source_reasons)
        forbidden = list(dict.fromkeys(forbidden))
        signature = derive_failure_signature(
            envelope_id=request.envelope.envelope_id,
            predicted_files=request.predicted_files,
            predicted_symbols=request.predicted_symbols,
            failure_signature=request.failure_signature,
        )
        prior = self._failures.get(signature)

        if forbidden:
            conditions = _empty_merge_conditions()
            return RepairControllerResult(
                disposition=RepairControllerDisposition.REJECTED,
                selected_tier=None,
                merge_disposition=MergeDisposition.REJECTED,
                merge_conditions=conditions,
                reason_codes=tuple(forbidden),
                diagnostic_reused=False,
                model_call_count=self._model_call_count,
                engine_run_count=self._engine_run_count,
                source_edit_disposition=(
                    "not_admitted" if request.source_edit_operator is not None else "not_source_edit"
                ),
                suffix_receipt_id=request.suffix_receipt_id,
            )

        if prior is not None:
            exhausted = prior.count >= self._max_identical_failures
            disposition = (
                RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
                if exhausted
                else RepairControllerDisposition.IDENTICAL_FAILURE_REUSED
            )
            updated = self._record_failure(
                signature,
                diagnostic_receipt_id=prior.diagnostic_receipt_id or request.diagnostic_receipt_id,
                now_ms=request.now_ms,
            )
            try:
                tier = self.select_tier(request)
                plan = self._build_plan(request, tier)
            except RepairControllerError:
                plan = None
                tier = None
            receipt = self._build_receipt(
                plan=plan,
                request=request,
                status=TerminalStatus.BLOCKED,
                failure_signature=signature,
                diagnostic_receipt_id=updated.diagnostic_receipt_id,
            )
            conditions = _empty_merge_conditions()
            return RepairControllerResult(
                disposition=disposition,
                selected_tier=tier,
                merge_disposition=MergeDisposition.NOT_ELIGIBLE,
                merge_conditions=conditions,
                reason_codes=(
                    disposition.value,
                    "identical_failure_reused_diagnosis",
                    "model_call_suppressed",
                ),
                plan=plan,
                receipt=receipt,
                diagnostic_reused=True,
                backoff_milliseconds=updated.backoff_milliseconds,
                model_call_count=self._model_call_count,
                engine_run_count=self._engine_run_count,
                suffix_receipt_id=request.suffix_receipt_id,
            )

        try:
            tier = self.select_tier(request)
            plan = self._build_plan(request, tier)
        except RepairControllerError:
            conditions = _empty_merge_conditions()
            return RepairControllerResult(
                disposition=RepairControllerDisposition.REJECTED,
                selected_tier=None,
                merge_disposition=MergeDisposition.REJECTED,
                merge_conditions=conditions,
                reason_codes=("tier_or_plan_rejected",),
                model_call_count=self._model_call_count,
                engine_run_count=self._engine_run_count,
                suffix_receipt_id=request.suffix_receipt_id,
            )

        engine_report: Mapping[str, Any] | None = None
        model_payload: Mapping[str, Any] = {}
        mutation_applied = False
        validation_pending = False
        source_edit_disposition = "not_source_edit"
        changed_paths = request.changed_paths or request.predicted_files
        reason_codes: list[str] = [tier.value, "exact_envelope_bound", "isolated_worktree_bound"]

        if request.delegate_to_engine and tier in {
            RepairTier.DETERMINISTIC,
            RepairTier.TEMPLATE_CONSTRAINED,
        }:
            report = self._delegate_engine(request)
            engine_report = None if report is None else report.to_dict()
            reason_codes.append("engine_delegated")
            if report is not None:
                reason_codes.append("no_second_repair_engine")

        if tier is RepairTier.MODEL_ASSISTED_BOUNDED:
            model_payload = self._call_model(plan, request)
            reason_codes.append("decision_runtime_required")
            if self._model_invoker is None:
                reason_codes.append("model_invoker_absent")

        if operator is not None and request.apply_source_edit:
            applied = self._apply_source_edit(request, operator)
            receipts = list(applied.get("receipts") or ())
            first = receipts[0] if receipts else {}
            source_edit_disposition = str(
                first.get("source_edit_disposition") or applied.get("source_edit_disposition") or ""
            ) or "validation_pending"
            mutation_applied = bool(first.get("mutation_applied"))
            validation_pending = True
            reason_codes.append("source_edit_validation_pending")
            if not applied.get("passed"):
                reason_codes.append("source_edit_not_complete")

        diagnostic_id = (
            str(model_payload.get("diagnostic_receipt_id") or "")
            or request.diagnostic_receipt_id
        )
        failed = request.failed or bool(model_payload.get("failed"))
        if failed:
            record = self._record_failure(
                signature,
                diagnostic_receipt_id=diagnostic_id,
                now_ms=request.now_ms,
            )
            receipt = self._build_receipt(
                plan=plan,
                request=request,
                status=TerminalStatus.FAILED,
                changed_paths=(),
                failure_signature=signature,
                diagnostic_receipt_id=record.diagnostic_receipt_id,
                rollback_receipt_id=request.rollback_plan_id,
            )
            conditions = evaluate_merge_conjunction(
                policy=request.policy,
                envelope=request.envelope,
                plan=plan,
                changed_paths=(),
                changed_line_count=0,
                validation_receipt_ids=request.validation_receipt_ids,
                proof_receipt_ids=request.proof_receipt_ids,
                terminal_status=TerminalStatus.FAILED,
                source_edit_validation_pending=validation_pending,
                forbidden_reasons=(),
            )
            return RepairControllerResult(
                disposition=RepairControllerDisposition.ROLLBACK_REQUIRED,
                selected_tier=tier,
                merge_disposition=merge_disposition_for(
                    conditions,
                    risk_class=request.envelope.risk_assessment.risk_class,
                    forbidden_reasons=(),
                ),
                merge_conditions=conditions,
                reason_codes=(*reason_codes, "rollback_required", "failure_recorded"),
                plan=plan,
                receipt=receipt,
                engine_report=engine_report,
                diagnostic_reused=False,
                backoff_milliseconds=record.backoff_milliseconds,
                model_call_count=self._model_call_count,
                engine_run_count=self._engine_run_count,
                source_edit_disposition=source_edit_disposition,
                mutation_applied=False,
                validation_pending=validation_pending,
                suffix_receipt_id=request.suffix_receipt_id,
            )

        if validation_pending or not request.validation_receipt_ids:
            status = TerminalStatus.PENDING
            receipt_changed: tuple[str, ...] = ()
            merge_changed: tuple[str, ...] = ()
            merge_lines = 0
        else:
            status = TerminalStatus.SUCCEEDED
            receipt_changed = tuple(changed_paths)
            merge_changed = tuple(changed_paths)
            merge_lines = request.changed_line_count
        receipt = self._build_receipt(
            plan=plan,
            request=request,
            status=status,
            changed_paths=receipt_changed,
            diagnostic_receipt_id=diagnostic_id,
        )
        conditions = evaluate_merge_conjunction(
            policy=request.policy,
            envelope=request.envelope,
            plan=plan,
            changed_paths=merge_changed,
            changed_line_count=merge_lines,
            validation_receipt_ids=request.validation_receipt_ids,
            proof_receipt_ids=request.proof_receipt_ids,
            terminal_status=status,
            source_edit_validation_pending=validation_pending,
            forbidden_reasons=(),
        )
        if validation_pending:
            disposition = RepairControllerDisposition.SOURCE_EDIT_VALIDATION_PENDING
        elif tier is RepairTier.MODEL_ASSISTED_BOUNDED:
            disposition = RepairControllerDisposition.MODEL_ASSISTED_DELEGATED
        else:
            disposition = RepairControllerDisposition.ENGINE_DELEGATED
        return RepairControllerResult(
            disposition=disposition,
            selected_tier=tier,
            merge_disposition=merge_disposition_for(
                conditions,
                risk_class=request.envelope.risk_assessment.risk_class,
                forbidden_reasons=(),
            ),
            merge_conditions=conditions,
            reason_codes=tuple(reason_codes),
            plan=plan,
            receipt=receipt,
            engine_report=engine_report,
            model_call_count=self._model_call_count,
            engine_run_count=self._engine_run_count,
            source_edit_disposition=source_edit_disposition,
            mutation_applied=mutation_applied,
            validation_pending=validation_pending,
            suffix_receipt_id=request.suffix_receipt_id,
        )


__all__ = [
    "AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE",
    "ENGINE_AUTHORITY",
    "LOW_RISK_MERGE_CONDITIONS",
    "MergeDisposition",
    "RepairControllerDisposition",
    "RepairControllerError",
    "RepairControllerRequest",
    "RepairControllerResult",
    "RepairFailureRecord",
    "SELF_PROTECTING_PATHS",
    "AutonomousRepairController",
    "classify_forbidden_paths",
    "derive_failure_signature",
    "evaluate_merge_conjunction",
    "merge_disposition_for",
    "path_is_self_edit",
    "path_is_sibling_repository",
    "path_is_validator_policy_key",
    "select_repair_tier",
]
