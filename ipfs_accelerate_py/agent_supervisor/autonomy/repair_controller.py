# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Bounded facade over the existing autonomous-repair engine.

``AutonomousRepairController@1`` selects a repair tier, binds one exact
envelope, and rejects scope escape, self-edits, and validator/policy-key
mutation.  It does not replace ``AutonomousRepairEngine``, admit source
bytes, issue a ``DecisionRuntime`` permit, or authorize merge.

Admission and mutation stay with the existing engine, materializer, and
decision-runtime authorities.  Identical failures reuse the stored diagnosis
and back off without a second model call.  Autonomous merge is the
conjunction of every stated low-risk condition; R3 remains proposal-only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Protocol

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
REPAIR_CONTROLLER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-controller-result@1"
)
REPAIR_FAILURE_MEMORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/repair-failure-memory@1"
)

ENGINE_INTERFACE: Final[str] = AUTONOMOUS_REPAIR_INTERFACE
SELF_EDIT_PATH: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
)
PROTECTED_AUTHORITY_PATHS: Final[tuple[str, ...]] = (
    SELF_EDIT_PATH,
    "ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",
    "ipfs_accelerate_py/agent_supervisor/autonomous_repair",
)
VALIDATOR_POLICY_KEY_PATH_MARKERS: Final[tuple[str, ...]] = (
    "authority_policy",
    "policy_key",
    "signing_key",
    "trusted_key",
    "validator_policy",
    "promotion_policy",
)
VALIDATOR_POLICY_KEY_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "authority_policy",
        "policy_key",
        "promotion_rules",
        "signing_keys",
        "trusted_keys",
        "validator_policy",
    }
)
LOW_RISK_MERGE_CONDITIONS: Final[tuple[str, ...]] = (
    "risk_at_most_r2",
    "reversible",
    "autonomy_level_permits_execute",
    "policy_allows_level",
    "autonomous_merge_enabled",
    "exact_paths_within_envelope",
    "exact_symbols_within_envelope",
    "no_scope_escape",
    "no_self_edit",
    "no_protected_authority_path",
    "no_validator_policy_key_mutation",
    "isolated_worktree_bound",
    "predetermined_tests_satisfied",
    "predetermined_proofs_satisfied",
    "rollback_plan_bound",
    "sufficient_context",
    "bounded_patch_envelope",
    "validation_receipts_current",
    "terminal_succeeded",
    "not_identical_failure_backoff",
)
DEFAULT_BACKOFF_MS: Final[int] = 1_000
MAX_BACKOFF_MS: Final[int] = 60_000
MAX_IDENTICAL_FAILURES: Final[int] = 3
MAX_CHANGED_FILES: Final[int] = 10_000
MAX_CHANGED_LINES: Final[int] = 1_000_000
_R2_OR_LOWER: Final[frozenset[RiskClass]] = frozenset(
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
    ADMITTED = "admitted"
    EXECUTED = "executed"
    REJECTED = "rejected"
    BACKOFF = "backoff"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    PROPOSAL_ONLY = "proposal_only"
    MERGE_ELIGIBLE = "merge_eligible"
    ROLLED_BACK = "rolled_back"


class RepairMergeDisposition(str, Enum):
    AUTONOMOUS_MERGE = "autonomous_merge"
    PROPOSAL_ONLY = "proposal_only"
    HUMAN_REQUIRED = "human_required"
    REJECTED = "rejected"
    INELIGIBLE = "ineligible"


class RepairEffectBoundary(str, Enum):
    FILE_MUTATION = "file_mutation"
    MERGE = "merge"


class RepairEffectGate(Protocol):
    """Narrow DecisionRuntime adapter; this facade never issues a permit."""

    def admits(
        self,
        boundary: str,
        *,
        envelope_id: str,
        plan_id: str,
    ) -> bool:
        """Return True only when DecisionRuntime already admitted the effect."""


class RepairModelInvoker(Protocol):
    """Optional model-assisted callback; identical failures must not invoke it."""

    def __call__(self, plan: AutonomousRepairPlan) -> Mapping[str, Any]:
        """Return a bounded diagnosis mapping, never a prompt or transcript."""


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


def _path(value: Any, name: str) -> str:
    result = _identifier(value, name)
    parsed = PurePosixPath(result)
    if "\\" in result or parsed.is_absolute() or ".." in parsed.parts or result in {".", ""}:
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
    normalized = tuple(_path(item, name) for item in identifiers)
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


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RepairControllerError(f"{name} must be an integer of at least {minimum}")
    if maximum is not None and value > maximum:
        raise RepairControllerError(f"{name} exceeds its bound")
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


def path_is_within(path: str, prefixes: Sequence[str]) -> bool:
    """Return True when ``path`` equals or is a descendant of one prefix."""

    candidate_path = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or candidate_path.is_absolute()
        or ".." in candidate_path.parts
        or path in {".", ""}
    ):
        return False
    candidate = candidate_path.as_posix()
    for prefix in prefixes:
        root_path = PurePosixPath(prefix)
        if (
            not prefix
            or "\\" in prefix
            or root_path.is_absolute()
            or ".." in root_path.parts
        ):
            continue
        root = root_path.as_posix().rstrip("/")
        if candidate == root or candidate.startswith(root + "/"):
            return True
    return False


def _normalized_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(".", "_")


def is_self_edit_path(path: str) -> bool:
    return path_is_within(path, (SELF_EDIT_PATH,))


def is_protected_authority_path(path: str) -> bool:
    return path_is_within(path, PROTECTED_AUTHORITY_PATHS)


def is_validator_policy_key_path(path: str) -> bool:
    token = _normalized_token(path)
    return any(marker in token for marker in VALIDATOR_POLICY_KEY_PATH_MARKERS)


def is_validator_policy_key_symbol(symbol: str) -> bool:
    token = _normalized_token(symbol)
    return token in VALIDATOR_POLICY_KEY_SYMBOLS or any(
        token.endswith("_" + marker) or marker in token
        for marker in VALIDATOR_POLICY_KEY_SYMBOLS
    )


def classify_scope_violations(
    paths: Sequence[str],
    symbols: Sequence[str],
    *,
    allowed_paths: Sequence[str],
    allowed_symbols: Sequence[str],
    forbidden_symbols: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return closed reason codes for envelope, self-edit, and key mutations."""

    reasons: list[str] = []
    allowed_symbol_set = set(allowed_symbols)
    forbidden_symbol_set = set(forbidden_symbols) | set(VALIDATOR_POLICY_KEY_SYMBOLS)
    for path in paths:
        if allowed_paths and not path_is_within(path, allowed_paths):
            reasons.append("scope_escape")
        if is_self_edit_path(path):
            reasons.append("self_edit")
        if is_protected_authority_path(path):
            reasons.append("protected_authority_path")
        if is_validator_policy_key_path(path):
            reasons.append("validator_policy_key_mutation")
    for symbol in symbols:
        if allowed_symbol_set and symbol not in allowed_symbol_set:
            reasons.append("symbol_escape")
        if symbol in forbidden_symbol_set or is_validator_policy_key_symbol(symbol):
            reasons.append("validator_policy_key_mutation")
    unique: list[str] = []
    seen: set[str] = set()
    for item in reasons:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)


def select_repair_tier(
    *,
    requested: RepairTier | str | None,
    template_id: str = "",
    model_assisted_requested: bool = False,
) -> RepairTier:
    """Choose the cheapest admissible tier.  Explicit requests are honored."""

    if requested is not None:
        return _enum(requested, RepairTier, "repair_tier")
    if template_id:
        return RepairTier.TEMPLATE_CONSTRAINED
    if model_assisted_requested:
        return RepairTier.MODEL_ASSISTED_BOUNDED
    return RepairTier.DETERMINISTIC


def _empty_merge_conditions() -> dict[str, bool]:
    return {name: False for name in LOW_RISK_MERGE_CONDITIONS}


@dataclass(frozen=True)
class RepairFailureRecord:
    """Bounded diagnosis reused across identical failures."""

    failure_signature: str
    diagnostic_receipt_id: str
    count: int
    last_seen_ms: int
    backoff_milliseconds: int

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
            self, "last_seen_ms", _int(self.last_seen_ms, "last_seen_ms")
        )
        object.__setattr__(
            self,
            "backoff_milliseconds",
            _int(self.backoff_milliseconds, "backoff_milliseconds"),
        )

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": REPAIR_FAILURE_MEMORY_SCHEMA,
                "failure_signature": self.failure_signature,
                "diagnostic_receipt_id": self.diagnostic_receipt_id,
                "count": self.count,
                "last_seen_ms": self.last_seen_ms,
                "backoff_milliseconds": self.backoff_milliseconds,
            }
        )


@dataclass(frozen=True)
class RepairControllerRequest:
    """One envelope-bound repair attempt.  Never a prompt or source body."""

    envelope: AutonomyEnvelope
    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    worktree_id: str
    rollback_plan_id: str
    context_reference_ids: tuple[str, ...]
    policy: AutonomyPolicy | None = None
    repair_tier: RepairTier | None = None
    template_id: str = ""
    operation: str = ""
    forbidden_symbols: tuple[str, ...] = ()
    max_changed_files: int = 1
    max_changed_lines: int = 100
    changed_paths: tuple[str, ...] = ()
    required_test_ids: tuple[str, ...] = ()
    required_proof_ids: tuple[str, ...] = ()
    validation_receipt_ids: tuple[str, ...] = ()
    proof_receipt_ids: tuple[str, ...] = ()
    adversarial_assurance_receipt_ids: tuple[str, ...] = ()
    terminal_status: TerminalStatus = TerminalStatus.PENDING
    failure_signature: str = ""
    diagnostic_receipt_id: str = ""
    new_evidence_ids: tuple[str, ...] = ()
    source_edit_operator: Mapping[str, Any] | None = None
    sufficient_context: bool = True
    model_assisted_requested: bool = False
    allow_code_edit_materialize: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, AutonomyEnvelope):
            raise RepairControllerError("envelope must be an AutonomyEnvelope")
        if self.policy is not None and not isinstance(self.policy, AutonomyPolicy):
            raise RepairControllerError("policy must be an AutonomyPolicy or None")
        if self.policy is not None and self.policy.policy_id != self.envelope.policy_id:
            raise RepairControllerError("envelope policy_id does not match policy")
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
        object.__setattr__(
            self,
            "context_reference_ids",
            _identifiers(self.context_reference_ids, "context_reference_ids", required=True),
        )
        if self.repair_tier is not None:
            object.__setattr__(
                self, "repair_tier", _enum(self.repair_tier, RepairTier, "repair_tier")
            )
        object.__setattr__(
            self, "template_id", _identifier(self.template_id, "template_id", required=False)
        )
        object.__setattr__(
            self, "operation", _identifier(self.operation, "operation", required=False)
        )
        object.__setattr__(
            self,
            "forbidden_symbols",
            _identifiers(self.forbidden_symbols, "forbidden_symbols"),
        )
        object.__setattr__(
            self,
            "max_changed_files",
            _int(self.max_changed_files, "max_changed_files", minimum=1, maximum=MAX_CHANGED_FILES),
        )
        object.__setattr__(
            self,
            "max_changed_lines",
            _int(self.max_changed_lines, "max_changed_lines", minimum=1, maximum=MAX_CHANGED_LINES),
        )
        object.__setattr__(
            self, "changed_paths", _paths(self.changed_paths, "changed_paths", preserve_order=True)
        )
        envelope_tests = self.envelope.required_test_ids
        envelope_proofs = self.envelope.required_proof_ids
        object.__setattr__(
            self,
            "required_test_ids",
            _identifiers(self.required_test_ids or envelope_tests, "required_test_ids"),
        )
        object.__setattr__(
            self,
            "required_proof_ids",
            _identifiers(self.required_proof_ids or envelope_proofs, "required_proof_ids"),
        )
        for name in (
            "validation_receipt_ids",
            "proof_receipt_ids",
            "adversarial_assurance_receipt_ids",
            "new_evidence_ids",
        ):
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
        object.__setattr__(
            self,
            "terminal_status",
            _enum(self.terminal_status, TerminalStatus, "terminal_status"),
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
                raise RepairControllerError("source_edit_operator must be an object")
            if len(self.source_edit_operator) > MAX_MAPPING_ITEMS:
                raise RepairControllerError("source_edit_operator contains too many fields")
            _reject_forbidden_keys(self.source_edit_operator, "source_edit_operator")
        for name in (
            "sufficient_context",
            "model_assisted_requested",
            "allow_code_edit_materialize",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if len(self.predicted_files) > self.max_changed_files:
            raise RepairControllerError("predicted repair files exceed the patch envelope")

    @property
    def selected_tier(self) -> RepairTier:
        return select_repair_tier(
            requested=self.repair_tier,
            template_id=self.template_id,
            model_assisted_requested=self.model_assisted_requested,
        )

    @property
    def observed_paths(self) -> tuple[str, ...]:
        if self.changed_paths:
            return self.changed_paths
        return self.predicted_files


@dataclass(frozen=True)
class RepairControllerResult:
    """Closed outcome of one facade evaluation.  Never authorizes an effect."""

    disposition: RepairControllerDisposition
    merge_disposition: RepairMergeDisposition
    repair_tier: RepairTier
    reason_codes: tuple[str, ...]
    low_risk_conditions: Mapping[str, bool]
    plan: AutonomousRepairPlan | None = None
    receipt: AutonomousRepairReceipt | None = None
    engine_interface: str = ENGINE_INTERFACE
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    model_call_count: int = 0
    source_edit_admitted: bool = False
    mutation_applied: bool = False
    authorizes_effect: bool = False
    authorizes_merge: bool = False

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
            self, "repair_tier", _enum(self.repair_tier, RepairTier, "repair_tier")
        )
        object.__setattr__(
            self, "reason_codes", _identifiers(self.reason_codes, "reason_codes", required=True)
        )
        conditions = dict(self.low_risk_conditions)
        if set(conditions) != set(LOW_RISK_MERGE_CONDITIONS):
            raise RepairControllerError("low_risk_conditions must report every stated condition")
        frozen = {name: _bool(conditions[name], "low_risk_conditions") for name in LOW_RISK_MERGE_CONDITIONS}
        object.__setattr__(self, "low_risk_conditions", MappingProxyType(frozen))
        if self.plan is not None and not isinstance(self.plan, AutonomousRepairPlan):
            raise RepairControllerError("plan must be an AutonomousRepairPlan or None")
        if self.receipt is not None and not isinstance(self.receipt, AutonomousRepairReceipt):
            raise RepairControllerError("receipt must be an AutonomousRepairReceipt or None")
        object.__setattr__(
            self,
            "engine_interface",
            _identifier(self.engine_interface, "engine_interface"),
        )
        if self.engine_interface != ENGINE_INTERFACE:
            raise RepairControllerError("controller must retain the existing engine interface")
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
        for name in (
            "source_edit_admitted",
            "mutation_applied",
            "authorizes_effect",
            "authorizes_merge",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if self.authorizes_effect or self.authorizes_merge:
            raise RepairControllerError(
                "repair controller results cannot authorize effects or merge"
            )
        if self.receipt is not None and self.receipt.authorizes_merge:
            raise RepairControllerError(
                "repair receipts are evidence and cannot independently authorize merge"
            )
        payload = self.to_dict(include_identity=False)
        if len(canonical_json(payload).encode("utf-8")) > MAX_CANONICAL_RECORD_BYTES:
            raise RepairControllerError("repair controller result exceeds its bounded size")

    @property
    def result_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def merge_eligible(self) -> bool:
        return self.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REPAIR_CONTROLLER_RESULT_SCHEMA,
            "interface": AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
            "disposition": self.disposition.value,
            "merge_disposition": self.merge_disposition.value,
            "repair_tier": self.repair_tier.value,
            "reason_codes": list(self.reason_codes),
            "low_risk_conditions": dict(self.low_risk_conditions),
            "plan": None if self.plan is None else self.plan.to_dict(),
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "engine_interface": self.engine_interface,
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "model_call_count": self.model_call_count,
            "source_edit_admitted": self.source_edit_admitted,
            "mutation_applied": self.mutation_applied,
            "authorizes_effect": False,
            "authorizes_merge": False,
        }
        if include_identity:
            payload["result_id"] = self.result_id
        return payload


def _backoff_ms(count: int) -> int:
    shift = max(count - 1, 0)
    value = DEFAULT_BACKOFF_MS * (2**shift)
    return min(value, MAX_BACKOFF_MS)


def evaluate_low_risk_merge_conditions(
    request: RepairControllerRequest,
    *,
    violations: Sequence[str] = (),
    backoff: bool = False,
    terminal_status: TerminalStatus | None = None,
) -> dict[str, bool]:
    """Return the closed R2 conjunction.  Every stated condition is reported."""

    envelope = request.envelope
    risk = envelope.risk_assessment
    policy = request.policy
    status = terminal_status if terminal_status is not None else request.terminal_status
    observed = request.observed_paths
    allowed_paths = envelope.allowed_paths
    allowed_symbols = envelope.allowed_symbols
    conditions = _empty_merge_conditions()
    conditions["risk_at_most_r2"] = risk.risk_class in _R2_OR_LOWER
    conditions["reversible"] = bool(risk.reversible and envelope.reversible)
    conditions["autonomy_level_permits_execute"] = (
        envelope.autonomy_level.rank >= AutonomyLevel.EXECUTE_REVERSIBLE.rank
    )
    conditions["policy_allows_level"] = policy is None or policy.allows(
        envelope.autonomy_level, risk.risk_class
    )
    conditions["autonomous_merge_enabled"] = bool(
        policy is not None and policy.autonomous_merge_enabled
    )
    conditions["exact_paths_within_envelope"] = bool(allowed_paths) and all(
        path_is_within(path, allowed_paths) for path in observed
    )
    conditions["exact_symbols_within_envelope"] = bool(allowed_symbols) and all(
        symbol in allowed_symbols for symbol in request.predicted_symbols
    )
    conditions["no_scope_escape"] = "scope_escape" not in violations and "symbol_escape" not in violations
    conditions["no_self_edit"] = "self_edit" not in violations
    conditions["no_protected_authority_path"] = "protected_authority_path" not in violations
    conditions["no_validator_policy_key_mutation"] = (
        "validator_policy_key_mutation" not in violations
    )
    conditions["isolated_worktree_bound"] = bool(request.worktree_id)
    required_tests = request.required_test_ids
    conditions["predetermined_tests_satisfied"] = bool(required_tests) and set(
        required_tests
    ).issubset(request.validation_receipt_ids)
    required_proofs = request.required_proof_ids
    conditions["predetermined_proofs_satisfied"] = set(required_proofs).issubset(
        request.proof_receipt_ids
    )
    conditions["rollback_plan_bound"] = bool(request.rollback_plan_id)
    conditions["sufficient_context"] = bool(
        request.sufficient_context and request.context_reference_ids
    )
    conditions["bounded_patch_envelope"] = (
        0 < len(observed) <= request.max_changed_files
        and request.max_changed_lines > 0
        and all(path_is_within(path, request.predicted_files) for path in request.changed_paths)
        if request.changed_paths
        else 0 < len(request.predicted_files) <= request.max_changed_files
    )
    conditions["validation_receipts_current"] = bool(request.validation_receipt_ids)
    conditions["terminal_succeeded"] = status is TerminalStatus.SUCCEEDED
    conditions["not_identical_failure_backoff"] = not backoff
    return conditions


def merge_disposition_for(
    request: RepairControllerRequest,
    conditions: Mapping[str, bool],
    *,
    violations: Sequence[str] = (),
) -> RepairMergeDisposition:
    risk = request.envelope.risk_assessment.risk_class
    if violations:
        return RepairMergeDisposition.REJECTED
    if risk is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL:
        return RepairMergeDisposition.HUMAN_REQUIRED
    if risk is RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE:
        return RepairMergeDisposition.HUMAN_REQUIRED
    if all(conditions[name] for name in LOW_RISK_MERGE_CONDITIONS):
        if risk in _R2_OR_LOWER:
            return RepairMergeDisposition.AUTONOMOUS_MERGE
    if risk is RiskClass.R3_BOUNDED_REPOSITORY_MUTATION:
        return RepairMergeDisposition.PROPOSAL_ONLY
    return RepairMergeDisposition.INELIGIBLE


def _model_assisted_gaps(request: RepairControllerRequest) -> tuple[str, ...]:
    gaps: list[str] = []
    if not request.predicted_files:
        gaps.append("exact_files_required")
    if not request.predicted_symbols:
        gaps.append("exact_symbols_required")
    if not request.envelope.allowed_paths:
        gaps.append("bounded_patch_envelope_required")
    if not request.context_reference_ids or not request.sufficient_context:
        gaps.append("sufficient_context_required")
    if not request.worktree_id:
        gaps.append("isolated_worktree_required")
    if not request.required_test_ids:
        gaps.append("predetermined_tests_required")
    return tuple(gaps)


class AutonomousRepairController:
    """Tier-selection and scope facade.  The existing engine remains authority."""

    def __init__(
        self,
        *,
        engine: AutonomousRepairEngine | None = None,
        materializer: Any | None = None,
        effect_gate: RepairEffectGate | None = None,
        model_invoker: Callable[[AutonomousRepairPlan], Mapping[str, Any]] | None = None,
        max_identical_failures: int = MAX_IDENTICAL_FAILURES,
    ) -> None:
        if engine is not None and not callable(getattr(engine, "run", None)):
            raise RepairControllerError("engine must expose the existing run() admission surface")
        self._engine = engine
        self._materializer = materializer
        self._effect_gate = effect_gate
        self._model_invoker = model_invoker
        self._max_identical_failures = _int(
            max_identical_failures, "max_identical_failures", minimum=1, maximum=32
        )
        self._failures: dict[str, RepairFailureRecord] = {}
        self._model_call_count = 0

    @property
    def interface(self) -> str:
        return AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE

    @property
    def engine_interface(self) -> str:
        return ENGINE_INTERFACE

    @property
    def engine(self) -> AutonomousRepairEngine | None:
        return self._engine

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    def failure_record(self, failure_signature: str) -> RepairFailureRecord | None:
        return self._failures.get(failure_signature)

    def select_tier(self, request: RepairControllerRequest) -> RepairTier:
        if not isinstance(request, RepairControllerRequest):
            raise RepairControllerError("request must be a RepairControllerRequest")
        return request.selected_tier

    def admit_source_edit(
        self,
        operator: Mapping[str, Any] | AdmittedSourceEditOperator | None,
        *,
        predicted_path: str,
        allowed_paths: Sequence[str],
        allow_code_edit_materialize: bool = False,
    ) -> AdmittedSourceEditOperator:
        """Admit an exact source-edit operator through the existing authority.

        A policy flag cannot become admission.  This method never writes bytes.
        """

        if allow_code_edit_materialize:
            raise RepairControllerError("policy_flag_is_not_source_edit_admission")
        if operator is None:
            raise AdmittedSourceEditError("source_edit_operator_missing")
        admitted = (
            operator
            if isinstance(operator, AdmittedSourceEditOperator)
            else AdmittedSourceEditOperator.from_mapping(operator)
        )
        try:
            relative = _path(admitted.relative_path, "source_edit_relative_path")
            bound_predicted = _path(predicted_path, "predicted_path")
        except RepairControllerError as exc:
            raise AdmittedSourceEditError("source_edit_path_binding_mismatch") from exc
        if relative != bound_predicted:
            raise AdmittedSourceEditError("source_edit_path_binding_mismatch")
        if not allowed_paths or not path_is_within(relative, allowed_paths):
            raise RepairControllerError("source_edit_scope_escape")
        if is_self_edit_path(relative) or is_protected_authority_path(relative):
            raise RepairControllerError("source_edit_protected_authority_path")
        if is_validator_policy_key_path(relative):
            raise RepairControllerError("source_edit_validator_policy_key_mutation")
        return admitted

    def build_plan(self, request: RepairControllerRequest) -> AutonomousRepairPlan:
        tier = request.selected_tier
        allowed = request.envelope.allowed_paths or request.predicted_files
        return AutonomousRepairPlan(
            objective_id=request.envelope.objective_id,
            task_id=request.envelope.task_id,
            repair_tier=tier,
            predicted_files=request.predicted_files,
            predicted_symbols=request.predicted_symbols,
            patch_envelope_id=request.envelope.envelope_id,
            context_reference_ids=request.context_reference_ids,
            required_test_ids=request.required_test_ids,
            required_proof_ids=request.required_proof_ids,
            worktree_id=request.worktree_id,
            allowed_paths=allowed,
            forbidden_symbols=request.forbidden_symbols,
            rollback_plan_id=request.rollback_plan_id,
            risk_class=request.envelope.risk_assessment.risk_class,
            max_changed_files=request.max_changed_files,
            max_changed_lines=request.max_changed_lines,
        )

    def evaluate(
        self,
        request: RepairControllerRequest,
        *,
        now_ms: int = 0,
    ) -> RepairControllerResult:
        if not isinstance(request, RepairControllerRequest):
            raise RepairControllerError("request must be a RepairControllerRequest")
        now = _int(now_ms, "now_ms")
        tier = request.selected_tier
        observed = request.observed_paths
        violations = classify_scope_violations(
            observed,
            request.predicted_symbols,
            allowed_paths=request.envelope.allowed_paths,
            allowed_symbols=request.envelope.allowed_symbols,
            forbidden_symbols=request.forbidden_symbols,
        )
        reasons: list[str] = []
        diagnostic_reused = False
        backoff_ms = 0
        model_calls = 0
        source_edit_admitted = False
        mutation_applied = False
        plan: AutonomousRepairPlan | None = None
        receipt: AutonomousRepairReceipt | None = None
        terminal = request.terminal_status
        diagnostic_id = request.diagnostic_receipt_id
        failure_signature = request.failure_signature

        backoff_hit = False
        existing = self._failures.get(failure_signature) if failure_signature else None
        if existing is not None and not request.new_evidence_ids:
            backoff_hit = True
            diagnostic_reused = True
            diagnostic_id = existing.diagnostic_receipt_id or diagnostic_id
            next_count = existing.count + 1
            backoff_ms = _backoff_ms(next_count)
            self._failures[failure_signature] = RepairFailureRecord(
                failure_signature=failure_signature,
                diagnostic_receipt_id=diagnostic_id,
                count=next_count,
                last_seen_ms=now,
                backoff_milliseconds=backoff_ms,
            )
            reasons.append("identical_failure_reused_diagnosis")
            reasons.append("model_call_suppressed")
            if next_count > self._max_identical_failures:
                reasons.append("identical_failure_exhausted")
                disposition = RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
            else:
                reasons.append("identical_failure_backoff")
                disposition = RepairControllerDisposition.BACKOFF
        elif violations:
            reasons.extend(violations)
            disposition = RepairControllerDisposition.REJECTED
        elif request.allow_code_edit_materialize:
            reasons.append("policy_flag_is_not_source_edit_admission")
            disposition = RepairControllerDisposition.REJECTED
            terminal = TerminalStatus.FAILED
        elif tier is RepairTier.MODEL_ASSISTED_BOUNDED and _model_assisted_gaps(request):
            reasons.extend(_model_assisted_gaps(request))
            disposition = RepairControllerDisposition.REJECTED
        elif tier is RepairTier.TEMPLATE_CONSTRAINED and not request.template_id:
            reasons.append("template_id_required")
            disposition = RepairControllerDisposition.REJECTED
        elif not request.worktree_id:
            reasons.append("isolated_worktree_required")
            disposition = RepairControllerDisposition.REJECTED
        else:
            plan = self.build_plan(request)
            engine_calls = self._run_engine(request)
            model_calls += engine_calls
            if (
                tier is RepairTier.MODEL_ASSISTED_BOUNDED
                and self._model_invoker is not None
                and not backoff_hit
            ):
                payload = self._model_invoker(plan)
                if not isinstance(payload, Mapping):
                    raise RepairControllerError("model invoker must return a bounded object")
                _reject_forbidden_keys(payload, "model diagnosis")
                self._model_call_count += 1
                model_calls += 1
                diagnostic_id = _identifier(
                    payload.get("diagnostic_receipt_id") or diagnostic_id,
                    "diagnostic_receipt_id",
                    required=False,
                )
                failure_signature = _identifier(
                    payload.get("failure_signature") or failure_signature,
                    "failure_signature",
                    required=False,
                )
            source_edit_admitted, mutation_applied, source_reasons = self._admit_and_maybe_mutate(
                request, plan
            )
            reasons.extend(source_reasons)
            if (
                "source_edit_not_admitted" in source_reasons
                or "source_edit_operator_missing" in source_reasons
            ):
                disposition = RepairControllerDisposition.REJECTED
                terminal = TerminalStatus.FAILED
            else:
                if terminal is TerminalStatus.SUCCEEDED:
                    if request.envelope.risk_assessment.risk_class is (
                        RiskClass.R3_BOUNDED_REPOSITORY_MUTATION
                    ):
                        disposition = RepairControllerDisposition.PROPOSAL_ONLY
                    else:
                        disposition = RepairControllerDisposition.EXECUTED
                else:
                    disposition = RepairControllerDisposition.ADMITTED
                reasons.append(tier.value)
                reasons.append("engine_remains_authority")
                if source_edit_admitted:
                    reasons.append("source_edit_admitted")
                if mutation_applied:
                    reasons.append("mutation_applied_by_existing_materializer")

        conditions = evaluate_low_risk_merge_conditions(
            request,
            violations=violations,
            backoff=backoff_hit,
            terminal_status=terminal,
        )
        merge = merge_disposition_for(request, conditions, violations=violations)
        if (
            disposition
            in {
                RepairControllerDisposition.EXECUTED,
                RepairControllerDisposition.ADMITTED,
                RepairControllerDisposition.PROPOSAL_ONLY,
            }
            and merge is RepairMergeDisposition.AUTONOMOUS_MERGE
        ):
            if self._effect_gate is not None and plan is not None:
                if not self._effect_gate.admits(
                    RepairEffectBoundary.MERGE.value,
                    envelope_id=request.envelope.envelope_id,
                    plan_id=plan.plan_id,
                ):
                    merge = RepairMergeDisposition.INELIGIBLE
                    conditions = dict(conditions)
                    conditions["autonomous_merge_enabled"] = False
                    reasons.append("decision_runtime_merge_not_admitted")
            if merge is RepairMergeDisposition.AUTONOMOUS_MERGE:
                disposition = RepairControllerDisposition.MERGE_ELIGIBLE
                reasons.append("r2_merge_conjunction")
        elif (
            disposition
            in {
                RepairControllerDisposition.EXECUTED,
                RepairControllerDisposition.ADMITTED,
            }
            and merge is RepairMergeDisposition.PROPOSAL_ONLY
        ):
            disposition = RepairControllerDisposition.PROPOSAL_ONLY
            reasons.append("r3_proposal_only")

        if (
            disposition is RepairControllerDisposition.REJECTED
            and merge is RepairMergeDisposition.AUTONOMOUS_MERGE
        ):
            merge = RepairMergeDisposition.REJECTED
        elif disposition in {
            RepairControllerDisposition.BACKOFF,
            RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED,
        } and merge is RepairMergeDisposition.AUTONOMOUS_MERGE:
            merge = RepairMergeDisposition.INELIGIBLE

        if plan is not None:
            changed = request.changed_paths if request.changed_paths else ()
            receipt_status = terminal
            if disposition in {
                RepairControllerDisposition.REJECTED,
                RepairControllerDisposition.BACKOFF,
                RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED,
            }:
                receipt_status = TerminalStatus.FAILED
            elif (
                receipt_status is TerminalStatus.SUCCEEDED
                and not request.validation_receipt_ids
            ):
                receipt_status = TerminalStatus.PENDING
            receipt = AutonomousRepairReceipt(
                plan_id=plan.plan_id,
                envelope_id=request.envelope.envelope_id,
                terminal_status=receipt_status,
                changed_paths=changed,
                validation_receipt_ids=request.validation_receipt_ids,
                proof_receipt_ids=request.proof_receipt_ids,
                adversarial_assurance_receipt_ids=request.adversarial_assurance_receipt_ids,
                rollback_receipt_id=request.rollback_plan_id
                if disposition is RepairControllerDisposition.ROLLED_BACK
                else "",
                failure_signature=failure_signature,
                diagnostic_receipt_id=diagnostic_id,
                authorizes_merge=False,
            )

        if (
            not backoff_hit
            and failure_signature
            and terminal is TerminalStatus.FAILED
            and disposition is not RepairControllerDisposition.REJECTED
        ):
            self._failures[failure_signature] = RepairFailureRecord(
                failure_signature=failure_signature,
                diagnostic_receipt_id=diagnostic_id,
                count=1,
                last_seen_ms=now,
                backoff_milliseconds=_backoff_ms(1),
            )

        if not reasons:
            reasons.append(disposition.value)

        return RepairControllerResult(
            disposition=disposition,
            merge_disposition=merge,
            repair_tier=tier,
            reason_codes=tuple(reasons),
            low_risk_conditions=conditions,
            plan=plan,
            receipt=receipt,
            diagnostic_reused=diagnostic_reused,
            backoff_milliseconds=backoff_ms,
            model_call_count=model_calls,
            source_edit_admitted=source_edit_admitted,
            mutation_applied=mutation_applied,
        )

    def _run_engine(self, request: RepairControllerRequest) -> int:
        if self._engine is None:
            return 0
        work = {
            "work_id": request.envelope.task_id,
            "operation": request.operation or request.predicted_symbols[0],
            "path": request.predicted_files[0],
            "symbol": request.predicted_symbols[0],
            "write_paths": list(request.predicted_files),
            "domain": "agent_supervisor",
        }
        report = self._engine.run([work])
        if report is None:
            return 0
        if isinstance(report, Mapping):
            return _int(report.get("model_call_count") or 0, "engine_model_call_count")
        return _int(getattr(report, "model_call_count", 0) or 0, "engine_model_call_count")

    def _admit_and_maybe_mutate(
        self,
        request: RepairControllerRequest,
        plan: AutonomousRepairPlan,
    ) -> tuple[bool, bool, tuple[str, ...]]:
        if request.source_edit_operator is None and self._materializer is None:
            return False, False, ()
        if request.source_edit_operator is None:
            return False, False, ("source_edit_operator_missing", "source_edit_not_admitted")
        try:
            admitted = self.admit_source_edit(
                request.source_edit_operator,
                predicted_path=request.predicted_files[0],
                allowed_paths=request.envelope.allowed_paths,
                allow_code_edit_materialize=request.allow_code_edit_materialize,
            )
        except AdmittedSourceEditError as exc:
            code = str(exc) or "source_edit_not_admitted"
            compact = code.split(":", 1)[-1].replace(" ", "_")
            return False, False, (compact, "source_edit_not_admitted")
        except RepairControllerError as exc:
            return False, False, (str(exc), "source_edit_not_admitted")
        mutation_applied = False
        reasons = ["typed_admitted_source_edit_operator"]
        if self._materializer is not None:
            if self._effect_gate is not None and not self._effect_gate.admits(
                RepairEffectBoundary.FILE_MUTATION.value,
                envelope_id=request.envelope.envelope_id,
                plan_id=plan.plan_id,
            ):
                reasons.append("decision_runtime_file_mutation_not_admitted")
                return True, False, tuple(reasons)
            batch = self._materializer.materialize_plans(
                [
                    {
                        "plan_id": plan.plan_id,
                        "work_id": request.envelope.task_id,
                        "operation": request.operation or request.predicted_symbols[0],
                        "materialize_ready": True,
                        "preferred_path": admitted.relative_path,
                        "handler": request.predicted_symbols[0],
                        "source_edit_operator": dict(request.source_edit_operator),
                    }
                ]
            )
            receipts = []
            summary_applied = 0
            if isinstance(batch, Mapping):
                receipts = list(batch.get("receipts") or ())
                summary = batch.get("summary") or {}
                if isinstance(summary, Mapping):
                    summary_applied = int(summary.get("applied") or 0)
            mutation_applied = summary_applied > 0 or any(
                (isinstance(item, Mapping) and item.get("mutation_applied") is True)
                or getattr(item, "mutation_applied", False) is True
                for item in receipts
            )
            if mutation_applied:
                reasons.append("existing_materializer_applied_admitted_bytes")
        return True, mutation_applied, tuple(reasons)


__all__ = [
    "AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE",
    "ENGINE_INTERFACE",
    "LOW_RISK_MERGE_CONDITIONS",
    "PROTECTED_AUTHORITY_PATHS",
    "SELF_EDIT_PATH",
    "AdmittedSourceEditError",
    "AdmittedSourceEditOperator",
    "AutonomousRepairController",
    "AutonomousRepairEngine",
    "RepairControllerDisposition",
    "RepairControllerError",
    "RepairControllerRequest",
    "RepairControllerResult",
    "RepairEffectBoundary",
    "RepairFailureRecord",
    "RepairMergeDisposition",
    "classify_scope_violations",
    "evaluate_low_risk_merge_conditions",
    "merge_disposition_for",
    "path_is_within",
    "select_repair_tier",
]
