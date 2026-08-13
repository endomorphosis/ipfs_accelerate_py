"""Shadow execution and semantic differential contracts (SCG-008).

Accelerate owns execution projections for paired compressed/expanded shadow
runs and semantic differential comparison.  These contracts reference
canonical datasets artifact headers and verification-bundle identities; they
do not mint another receipt hierarchy or generic envelope.

Normative invariants (fail-closed):

* Text difference alone cannot classify failure.
* Expanded output is never marked ``accepted`` by construction (oracle /
  candidate only; never silently replaces the accepted patch).
* Simulated versus live provenance is unambiguous via
  ``ArtifactProvenance.execution_mode`` and per-attempt
  ``execution_mode``.

Closed comparative outcomes match the plan vocabulary exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence
import re
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
    validate_structured_value,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
    SemanticGovernorBaseError,
    reject_private_and_model_authority,
)

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

SEMANTIC_GOVERNOR_EXECUTION_INTERFACE: Final[str] = "SemanticGovernorExecution@1"
SCG_EXECUTION_CONTRACTS_EVIDENCE: Final[str] = "scg/execution-contracts@1"

SHADOW_EXECUTION_PLAN_INTERFACE: Final[str] = "ShadowExecutionPlan@1"
SHADOW_EXECUTION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "shadow-execution-plan@1"
)

SHADOW_EXECUTION_RESULT_INTERFACE: Final[str] = "ShadowExecutionResult@1"
SHADOW_EXECUTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "shadow-execution-result@1"
)

DIFFERENTIAL_PATCH_REPORT_INTERFACE: Final[str] = "DifferentialPatchReport@1"
DIFFERENTIAL_PATCH_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "differential-patch-report@1"
)

SEMANTIC_OUTCOME_COMPARISON_INTERFACE: Final[str] = "SemanticOutcomeComparison@1"
SEMANTIC_OUTCOME_COMPARISON_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "semantic-outcome-comparison@1"
)

PAIRED_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/paired-attempt@1"
)
COST_TIMING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/cost-timing@1"
)
VERIFICATION_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "verification-projection@1"
)

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_REASON_CODES: Final[int] = 256
MAX_EDIT_CLASSES: Final[int] = 256
MAX_CLASSIFICATION_BASES: Final[int] = 64
MAX_SELECTION_REASONS: Final[int] = 64

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_TASK_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.:/+-]{0,127}$"
)


class SemanticGovernorExecutionError(SemanticGovernorBaseError):
    """Raised when a shadow/differential execution contract is malformed."""


# ---------------------------------------------------------------------------
# Closed enumerations
# ---------------------------------------------------------------------------


class ComparativeOutcome(str, Enum):
    """Closed comparative outcomes (exactly ten; plan §5)."""

    EQUIVALENT_SUCCESS = "equivalent_success"
    COMPRESSED_BETTER = "compressed_better"
    EXPANDED_BETTER = "expanded_better"
    BOTH_VALID_DIFFERENT = "both_valid_different"
    COMPRESSED_FAILED_EXPANDED_SUCCEEDED = "compressed_failed_expanded_succeeded"
    COMPRESSED_SUCCEEDED_EXPANDED_FAILED = "compressed_succeeded_expanded_failed"
    BOTH_FAILED_SAME_REASON = "both_failed_same_reason"
    BOTH_FAILED_DIFFERENT_REASON = "both_failed_different_reason"
    VERIFICATION_INCONCLUSIVE = "verification_inconclusive"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class ShadowAttemptRole(str, Enum):
    """Paired attempt roles in a shadow plan."""

    COMPRESSED = "compressed"
    EXPANDED = "expanded"


class AttemptTerminalStatus(str, Enum):
    """Closed terminal status for one shadow attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    CANCELLED = "cancelled"
    EVALUATION_FAILED = "evaluation_failed"
    SKIPPED = "skipped"


class AcceptanceDisposition(str, Enum):
    """Acceptance projection; expanded is never ``accepted``."""

    NOT_ACCEPTED = "not_accepted"
    CANDIDATE_ONLY = "candidate_only"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    ACCEPTED = "accepted"


class OutcomeClassificationBasis(str, Enum):
    """Evidence bases admitted when classifying differential outcomes.

    ``text_diff`` may be recorded as an observation but alone can never
    classify failure.
    """

    VERIFICATION_RECEIPTS = "verification_receipts"
    PROOF_RECEIPTS = "proof_receipts"
    COUNTEREXAMPLE_RECEIPTS = "counterexample_receipts"
    AST_EDIT_CLASSES = "ast_edit_classes"
    INTERFACE_DIFF = "interface_diff"
    SIDE_EFFECT_DIFF = "side_effect_diff"
    EXCEPTION_CONTRACT_DIFF = "exception_contract_diff"
    SCHEMA_DIFF = "schema_diff"
    TEST_RESULT_DIFF = "test_result_diff"
    STATIC_ANALYSIS_DIFF = "static_analysis_diff"
    PERFORMANCE_DIFF = "performance_diff"
    ACCEPTANCE_MATRIX_DIFF = "acceptance_matrix_diff"
    HUMAN_REVIEW = "human_review"
    COST_TIMING = "cost_timing"
    TEXT_DIFF = "text_diff"


class SemanticEditClass(str, Enum):
    """Closed AST / structural edit classes for differential projection."""

    IDENTICAL = "identical"
    EQUIVALENT_REFORMAT = "equivalent_reformat"
    RENAME = "rename"
    REORDER = "reorder"
    ADD = "add"
    REMOVE = "remove"
    MODIFY_LOGIC = "modify_logic"
    INTERFACE_CHANGE = "interface_change"
    UNKNOWN = "unknown"


class ShadowSelectionReason(str, Enum):
    """Closed reasons a paired shadow evaluation was selected."""

    RISK_CLASS_MANDATORY = "risk_class_mandatory"
    CAPSULE_UNCERTAINTY = "capsule_uncertainty"
    NEW_ANALYZER = "new_analyzer"
    NEW_TASK_CLASS = "new_task_class"
    NEW_ROUTE = "new_route"
    TOKEN_SAVINGS_SAMPLE = "token_savings_sample"
    PROOF_CACHE_REUSE = "proof_cache_reuse"
    RECENT_OMISSION = "recent_omission"
    RANDOM_QUALITY_CONTROL = "random_quality_control"
    PROMOTION_EVALUATION = "promotion_evaluation"
    DEVELOPMENT_FULL_RATE = "development_full_rate"
    DISCLOSURE_FORBIDDEN_SKIP = "disclosure_forbidden_skip"


# Bases that count as non-text semantic/verification evidence.
_NON_TEXT_CLASSIFICATION_BASES: Final[frozenset[str]] = frozenset(
    item.value
    for item in OutcomeClassificationBasis
    if item is not OutcomeClassificationBasis.TEXT_DIFF
)

_FAILURE_LIKE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value,
        ComparativeOutcome.COMPRESSED_SUCCEEDED_EXPANDED_FAILED.value,
        ComparativeOutcome.BOTH_FAILED_SAME_REASON.value,
        ComparativeOutcome.BOTH_FAILED_DIFFERENT_REASON.value,
        ComparativeOutcome.EXPANDED_BETTER.value,
        ComparativeOutcome.COMPRESSED_BETTER.value,
    }
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise SemanticGovernorExecutionError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise SemanticGovernorExecutionError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise SemanticGovernorExecutionError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise SemanticGovernorExecutionError(
            f"{name} has unsupported value {value!r}"
        ) from exc


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise SemanticGovernorExecutionError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise SemanticGovernorExecutionError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _task_id(value: Any, name: str = "task_id") -> str:
    text = _text(value, name)
    if _TASK_ID_RE.fullmatch(text) is None:
        raise SemanticGovernorExecutionError(
            f"{name} must match {_TASK_ID_RE.pattern}"
        )
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise SemanticGovernorExecutionError(f"{name} must be a nonnegative integer")
    return value


def _int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise SemanticGovernorExecutionError(f"{name} must be an integer")
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SemanticGovernorExecutionError(f"{name} must be a boolean")
    return value


def _freeze_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_structured(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_structured(item) for item in value)
    return value


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_structured(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_structured(item) for item in value]
    return value


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise SemanticGovernorExecutionError(f"{name} must be a mapping")
    actual = set(data)
    if actual != fields:
        raise SemanticGovernorExecutionError(
            f"{name} fields must be exactly {sorted(fields)}, got {sorted(actual)}"
        )
    return dict(data)


def _require_structured(value: Any, name: str) -> Any:
    thawed = _thaw_structured(value)
    try:
        validate_structured_value(thawed, path=name)
    except Exception as exc:
        raise SemanticGovernorExecutionError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    try:
        reject_private_and_model_authority(thawed, path=name)
    except SemanticGovernorBaseError as exc:
        # Re-raise as execution error so consumers catch one closed type.
        raise SemanticGovernorExecutionError(str(exc)) from exc
    return thawed


def _mapping(value: Any, name: str, *, frozen: bool = True) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticGovernorExecutionError(f"{name} must be a mapping")
    result = _require_structured(dict(value), name)
    return _freeze_structured(result) if frozen else result


def _unique_sorted_enums(
    values: Iterable[Any],
    enum_type: type[Enum],
    name: str,
    *,
    max_items: int,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SemanticGovernorExecutionError(f"{name} must be a list")
    ordered = tuple(sorted(_enum(value, enum_type, name) for value in values))
    if len(ordered) > max_items:
        raise SemanticGovernorExecutionError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise SemanticGovernorExecutionError(f"{name} must not contain duplicates")
    return ordered


def _unique_sorted_tokens(values: Iterable[Any], name: str, *, max_items: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SemanticGovernorExecutionError(f"{name} must be a list")
    ordered = tuple(sorted(_token(value, name) for value in values))
    if len(ordered) > max_items:
        raise SemanticGovernorExecutionError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise SemanticGovernorExecutionError(f"{name} must not contain duplicates")
    return ordered


def _normalize_header(
    value: GovernorArtifactHeader | Mapping[str, Any],
    *,
    expected_kind: str,
) -> GovernorArtifactHeader:
    if isinstance(value, GovernorArtifactHeader):
        header = value
    elif isinstance(value, Mapping):
        header = GovernorArtifactHeader.from_dict(value)
    else:
        raise SemanticGovernorExecutionError(
            "header must be GovernorArtifactHeader or mapping"
        )
    if header.artifact_kind != expected_kind:
        raise SemanticGovernorExecutionError(
            f"header.artifact_kind must be {expected_kind!r}, "
            f"got {header.artifact_kind!r}"
        )
    return header


def non_text_classification_bases(
    bases: Sequence[str] | Iterable[str],
) -> frozenset[str]:
    """Return classification bases other than pure text difference."""

    return frozenset(bases) & _NON_TEXT_CLASSIFICATION_BASES


def assert_failure_classification_not_text_alone(
    bases: Sequence[str] | Iterable[str],
    *,
    failure_classified: bool,
    name: str = "classification_bases",
) -> None:
    """Reject failure classification supported only by text difference."""

    if not failure_classified:
        return
    if not non_text_classification_bases(bases):
        raise SemanticGovernorExecutionError(
            f"{name}: text difference alone cannot classify failure; "
            "require verification, structural, or human-review evidence"
        )


def assert_expanded_never_accepted(
    disposition: str,
    *,
    role: str,
    name: str = "acceptance_disposition",
) -> None:
    """Expanded shadow output is never accepted by construction."""

    if (
        role == ShadowAttemptRole.EXPANDED.value
        and disposition == AcceptanceDisposition.ACCEPTED.value
    ):
        raise SemanticGovernorExecutionError(
            f"{name}: expanded output is never marked accepted by construction"
        )


def comparative_outcomes() -> tuple[str, ...]:
    """Return the closed comparative-outcome vocabulary in declaration order."""

    return tuple(item.value for item in ComparativeOutcome)


def outcome_classification_bases() -> tuple[str, ...]:
    """Return admitted classification bases in declaration order."""

    return tuple(item.value for item in OutcomeClassificationBasis)


# ---------------------------------------------------------------------------
# Cost / timing projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostTimingProjection:
    """Bounded cost and timing projection (integer micros / ms; no floats)."""

    input_tokens: int
    output_tokens: int
    wall_time_ms: int
    model_spend_micros: int
    verification_time_ms: int = 0

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "input_tokens",
            "output_tokens",
            "wall_time_ms",
            "model_spend_micros",
            "verification_time_ms",
            "projection_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "input_tokens", _nonneg_int(self.input_tokens, "input_tokens")
        )
        object.__setattr__(
            self, "output_tokens", _nonneg_int(self.output_tokens, "output_tokens")
        )
        object.__setattr__(
            self, "wall_time_ms", _nonneg_int(self.wall_time_ms, "wall_time_ms")
        )
        object.__setattr__(
            self,
            "model_spend_micros",
            _nonneg_int(self.model_spend_micros, "model_spend_micros"),
        )
        object.__setattr__(
            self,
            "verification_time_ms",
            _nonneg_int(self.verification_time_ms, "verification_time_ms"),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": COST_TIMING_SCHEMA,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_time_ms": self.wall_time_ms,
            "model_spend_micros": self.model_spend_micros,
            "verification_time_ms": self.verification_time_ms,
        }

    @property
    def projection_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["projection_cid"] = self.projection_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostTimingProjection":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("projection_cid")
        if payload.pop("schema") != COST_TIMING_SCHEMA:
            raise SemanticGovernorExecutionError(
                "unsupported CostTimingProjection schema version"
            )
        result = cls(
            input_tokens=payload["input_tokens"],
            output_tokens=payload["output_tokens"],
            wall_time_ms=payload["wall_time_ms"],
            model_spend_micros=payload["model_spend_micros"],
            verification_time_ms=payload["verification_time_ms"],
        )
        if claimed != result.projection_cid:
            raise SemanticGovernorExecutionError(
                "CostTimingProjection projection_cid does not verify"
            )
        return result


# ---------------------------------------------------------------------------
# Verification projection (references; not a new receipt hierarchy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerificationProjection:
    """Projection onto canonical verification-bundle evidence.

    Does not re-encode receipts; binds the verification-bundle CID and closed
    boolean projections required for differential comparison.
    """

    verification_bundle_cid: str
    selected_tests_passed: bool | None
    full_suite_passed: bool | None
    proofs_passed: bool | None
    static_checks_passed: bool | None
    counterexample_present: bool
    acceptance_matrix_satisfied: bool
    production_eligible: bool

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "verification_bundle_cid",
            "selected_tests_passed",
            "full_suite_passed",
            "proofs_passed",
            "static_checks_passed",
            "counterexample_present",
            "acceptance_matrix_satisfied",
            "production_eligible",
            "projection_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verification_bundle_cid",
            _cid(self.verification_bundle_cid, "verification_bundle_cid"),
        )
        for field_name in (
            "selected_tests_passed",
            "full_suite_passed",
            "proofs_passed",
            "static_checks_passed",
        ):
            value = getattr(self, field_name)
            if value is not None and type(value) is not bool:
                raise SemanticGovernorExecutionError(
                    f"{field_name} must be a boolean or null"
                )
        object.__setattr__(
            self,
            "counterexample_present",
            _bool(self.counterexample_present, "counterexample_present"),
        )
        object.__setattr__(
            self,
            "acceptance_matrix_satisfied",
            _bool(self.acceptance_matrix_satisfied, "acceptance_matrix_satisfied"),
        )
        object.__setattr__(
            self,
            "production_eligible",
            _bool(self.production_eligible, "production_eligible"),
        )
        # Production eligibility requires a satisfied acceptance matrix.
        if self.production_eligible and not self.acceptance_matrix_satisfied:
            raise SemanticGovernorExecutionError(
                "production_eligible requires acceptance_matrix_satisfied"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": VERIFICATION_PROJECTION_SCHEMA,
            "verification_bundle_cid": self.verification_bundle_cid,
            "selected_tests_passed": self.selected_tests_passed,
            "full_suite_passed": self.full_suite_passed,
            "proofs_passed": self.proofs_passed,
            "static_checks_passed": self.static_checks_passed,
            "counterexample_present": self.counterexample_present,
            "acceptance_matrix_satisfied": self.acceptance_matrix_satisfied,
            "production_eligible": self.production_eligible,
        }

    @property
    def projection_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["projection_cid"] = self.projection_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationProjection":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("projection_cid")
        if payload.pop("schema") != VERIFICATION_PROJECTION_SCHEMA:
            raise SemanticGovernorExecutionError(
                "unsupported VerificationProjection schema version"
            )
        result = cls(
            verification_bundle_cid=payload["verification_bundle_cid"],
            selected_tests_passed=payload["selected_tests_passed"],
            full_suite_passed=payload["full_suite_passed"],
            proofs_passed=payload["proofs_passed"],
            static_checks_passed=payload["static_checks_passed"],
            counterexample_present=payload["counterexample_present"],
            acceptance_matrix_satisfied=payload["acceptance_matrix_satisfied"],
            production_eligible=payload["production_eligible"],
        )
        if claimed != result.projection_cid:
            raise SemanticGovernorExecutionError(
                "VerificationProjection projection_cid does not verify"
            )
        return result


# ---------------------------------------------------------------------------
# Paired attempt record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairedAttemptRecord:
    """One compressed or expanded shadow attempt with cost and verification.

    ``execution_mode`` makes simulated/live/replay provenance unambiguous at
    the attempt boundary (in addition to the durable header provenance).
    """

    role: ShadowAttemptRole | str
    execution_mode: ExecutionMode | str
    context_pack_cid: str
    route_id: str
    attempt_status: AttemptTerminalStatus | str
    acceptance_disposition: AcceptanceDisposition | str
    cost_timing: CostTimingProjection
    verification: VerificationProjection
    patch_cid: str | None = None
    worktree_id: str | None = None
    failure_reason_codes: Sequence[str] = ()
    notes: str | None = None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "role",
            "execution_mode",
            "context_pack_cid",
            "route_id",
            "attempt_status",
            "acceptance_disposition",
            "cost_timing",
            "verification",
            "patch_cid",
            "worktree_id",
            "failure_reason_codes",
            "notes",
            "attempt_cid",
        }
    )

    def __post_init__(self) -> None:
        role = _enum(self.role, ShadowAttemptRole, "role")
        object.__setattr__(self, "role", role)
        mode = _enum(self.execution_mode, ExecutionMode, "execution_mode")
        object.__setattr__(self, "execution_mode", mode)
        object.__setattr__(
            self, "context_pack_cid", _cid(self.context_pack_cid, "context_pack_cid")
        )
        object.__setattr__(self, "route_id", _token(self.route_id, "route_id"))
        status = _enum(self.attempt_status, AttemptTerminalStatus, "attempt_status")
        object.__setattr__(self, "attempt_status", status)
        disposition = _enum(
            self.acceptance_disposition,
            AcceptanceDisposition,
            "acceptance_disposition",
        )
        object.__setattr__(self, "acceptance_disposition", disposition)
        if not isinstance(self.cost_timing, CostTimingProjection):
            raise SemanticGovernorExecutionError(
                "cost_timing must be CostTimingProjection"
            )
        if not isinstance(self.verification, VerificationProjection):
            raise SemanticGovernorExecutionError(
                "verification must be VerificationProjection"
            )
        object.__setattr__(self, "patch_cid", _optional_cid(self.patch_cid, "patch_cid"))
        object.__setattr__(
            self, "worktree_id", _optional_text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self,
            "failure_reason_codes",
            _unique_sorted_tokens(
                list(self.failure_reason_codes),
                "failure_reason_codes",
                max_items=MAX_REASON_CODES,
            ),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))

        # Expanded is oracle/candidate only — never accepted, never production.
        assert_expanded_never_accepted(disposition, role=role)
        if role == ShadowAttemptRole.EXPANDED.value:
            if disposition not in {
                AcceptanceDisposition.NOT_ACCEPTED.value,
                AcceptanceDisposition.CANDIDATE_ONLY.value,
                AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value,
            }:
                raise SemanticGovernorExecutionError(
                    "expanded acceptance_disposition must be candidate-only, "
                    "not_accepted, or human_review_required"
                )
            if self.verification.production_eligible:
                raise SemanticGovernorExecutionError(
                    "expanded attempt cannot be production_eligible"
                )
            if disposition == AcceptanceDisposition.CANDIDATE_ONLY.value:
                pass  # expected happy path for a valid expanded oracle

        # Simulated provenance is unambiguous and non-accepting for production.
        if mode == ExecutionMode.SIMULATED.value:
            if disposition == AcceptanceDisposition.ACCEPTED.value:
                raise SemanticGovernorExecutionError(
                    "simulated attempt cannot claim acceptance_disposition=accepted"
                )
            if self.verification.production_eligible:
                raise SemanticGovernorExecutionError(
                    "simulated attempt cannot be production_eligible"
                )

        if (
            disposition == AcceptanceDisposition.ACCEPTED.value
            and not self.verification.production_eligible
        ):
            raise SemanticGovernorExecutionError(
                "acceptance_disposition=accepted requires production_eligible"
            )

        if status == AttemptTerminalStatus.FAILED.value and not self.failure_reason_codes:
            raise SemanticGovernorExecutionError(
                "failed attempt requires at least one failure_reason_code"
            )

        if (
            status == AttemptTerminalStatus.SUCCEEDED.value
            and disposition == AcceptanceDisposition.ACCEPTED.value
            and mode != ExecutionMode.LIVE.value
        ):
            raise SemanticGovernorExecutionError(
                "accepted succeeded attempts require live execution_mode"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PAIRED_ATTEMPT_SCHEMA,
            "role": self.role,
            "execution_mode": self.execution_mode,
            "context_pack_cid": self.context_pack_cid,
            "route_id": self.route_id,
            "attempt_status": self.attempt_status,
            "acceptance_disposition": self.acceptance_disposition,
            "cost_timing": self.cost_timing.identity_payload(),
            "verification": self.verification.identity_payload(),
            "patch_cid": self.patch_cid,
            "worktree_id": self.worktree_id,
            "failure_reason_codes": list(self.failure_reason_codes),
            "notes": self.notes,
        }

    @property
    def attempt_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIRED_ATTEMPT_SCHEMA,
            "role": self.role,
            "execution_mode": self.execution_mode,
            "context_pack_cid": self.context_pack_cid,
            "route_id": self.route_id,
            "attempt_status": self.attempt_status,
            "acceptance_disposition": self.acceptance_disposition,
            "cost_timing": self.cost_timing.to_dict(),
            "verification": self.verification.to_dict(),
            "patch_cid": self.patch_cid,
            "worktree_id": self.worktree_id,
            "failure_reason_codes": list(self.failure_reason_codes),
            "notes": self.notes,
            "attempt_cid": self.attempt_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PairedAttemptRecord":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("attempt_cid")
        if payload.pop("schema") != PAIRED_ATTEMPT_SCHEMA:
            raise SemanticGovernorExecutionError(
                "unsupported PairedAttemptRecord schema version"
            )
        cost_raw = payload["cost_timing"]
        ver_raw = payload["verification"]
        if not isinstance(cost_raw, Mapping) or not isinstance(ver_raw, Mapping):
            raise SemanticGovernorExecutionError(
                "cost_timing and verification must be mappings"
            )
        result = cls(
            role=payload["role"],
            execution_mode=payload["execution_mode"],
            context_pack_cid=payload["context_pack_cid"],
            route_id=payload["route_id"],
            attempt_status=payload["attempt_status"],
            acceptance_disposition=payload["acceptance_disposition"],
            cost_timing=CostTimingProjection.from_dict(cost_raw),
            verification=VerificationProjection.from_dict(ver_raw),
            patch_cid=payload["patch_cid"],
            worktree_id=payload["worktree_id"],
            failure_reason_codes=payload["failure_reason_codes"],
            notes=payload["notes"],
        )
        if claimed != result.attempt_cid:
            raise SemanticGovernorExecutionError(
                "PairedAttemptRecord attempt_cid does not verify"
            )
        return result


# ---------------------------------------------------------------------------
# ShadowExecutionPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowExecutionPlan:
    """Bounded plan for a paired compressed/expanded shadow evaluation.

    The expanded run is an oracle/candidate only
    (``expanded_is_oracle_candidate_only`` is required true) and executes in
    an isolated evaluation worktree.
    """

    header: GovernorArtifactHeader
    task_id: str
    audit_policy_cid: str
    compressed_context_pack_cid: str
    expanded_context_pack_cid: str
    compressed_route_id: str
    expanded_route_id: str
    selection_reasons: Sequence[str]
    max_wall_time_ms: int
    max_model_spend_micros: int
    max_expansion_token_budget: int
    isolated_evaluation_worktree_required: bool = True
    expanded_is_oracle_candidate_only: bool = True
    allow_external_expanded_disclosure: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "header",
            "task_id",
            "audit_policy_cid",
            "compressed_context_pack_cid",
            "expanded_context_pack_cid",
            "compressed_route_id",
            "expanded_route_id",
            "selection_reasons",
            "max_wall_time_ms",
            "max_model_spend_micros",
            "max_expansion_token_budget",
            "isolated_evaluation_worktree_required",
            "expanded_is_oracle_candidate_only",
            "allow_external_expanded_disclosure",
            "metadata",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "header",
            _normalize_header(self.header, expected_kind="shadow_execution_plan"),
        )
        object.__setattr__(self, "task_id", _task_id(self.task_id))
        object.__setattr__(
            self, "audit_policy_cid", _cid(self.audit_policy_cid, "audit_policy_cid")
        )
        object.__setattr__(
            self,
            "compressed_context_pack_cid",
            _cid(self.compressed_context_pack_cid, "compressed_context_pack_cid"),
        )
        object.__setattr__(
            self,
            "expanded_context_pack_cid",
            _cid(self.expanded_context_pack_cid, "expanded_context_pack_cid"),
        )
        object.__setattr__(
            self,
            "compressed_route_id",
            _token(self.compressed_route_id, "compressed_route_id"),
        )
        object.__setattr__(
            self,
            "expanded_route_id",
            _token(self.expanded_route_id, "expanded_route_id"),
        )
        object.__setattr__(
            self,
            "selection_reasons",
            _unique_sorted_enums(
                list(self.selection_reasons),
                ShadowSelectionReason,
                "selection_reasons",
                max_items=MAX_SELECTION_REASONS,
            ),
        )
        if not self.selection_reasons:
            raise SemanticGovernorExecutionError(
                "selection_reasons must contain at least one reason"
            )
        object.__setattr__(
            self,
            "max_wall_time_ms",
            _nonneg_int(self.max_wall_time_ms, "max_wall_time_ms"),
        )
        object.__setattr__(
            self,
            "max_model_spend_micros",
            _nonneg_int(self.max_model_spend_micros, "max_model_spend_micros"),
        )
        object.__setattr__(
            self,
            "max_expansion_token_budget",
            _nonneg_int(self.max_expansion_token_budget, "max_expansion_token_budget"),
        )
        object.__setattr__(
            self,
            "isolated_evaluation_worktree_required",
            _bool(
                self.isolated_evaluation_worktree_required,
                "isolated_evaluation_worktree_required",
            ),
        )
        object.__setattr__(
            self,
            "expanded_is_oracle_candidate_only",
            _bool(
                self.expanded_is_oracle_candidate_only,
                "expanded_is_oracle_candidate_only",
            ),
        )
        object.__setattr__(
            self,
            "allow_external_expanded_disclosure",
            _bool(
                self.allow_external_expanded_disclosure,
                "allow_external_expanded_disclosure",
            ),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        if not self.isolated_evaluation_worktree_required:
            raise SemanticGovernorExecutionError(
                "isolated_evaluation_worktree_required must be true"
            )
        if not self.expanded_is_oracle_candidate_only:
            raise SemanticGovernorExecutionError(
                "expanded_is_oracle_candidate_only must be true by construction"
            )
        if (
            ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value
            in self.selection_reasons
            and self.allow_external_expanded_disclosure
        ):
            raise SemanticGovernorExecutionError(
                "disclosure_forbidden_skip cannot allow external expanded disclosure"
            )
        # Header context pack should bind the compressed pack for the plan.
        if self.header.context_pack_cid != self.compressed_context_pack_cid:
            raise SemanticGovernorExecutionError(
                "header.context_pack_cid must equal compressed_context_pack_cid"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_EXECUTION_PLAN_SCHEMA,
            "interface_id": SHADOW_EXECUTION_PLAN_INTERFACE,
            "header": self.header.identity_payload(),
            "task_id": self.task_id,
            "audit_policy_cid": self.audit_policy_cid,
            "compressed_context_pack_cid": self.compressed_context_pack_cid,
            "expanded_context_pack_cid": self.expanded_context_pack_cid,
            "compressed_route_id": self.compressed_route_id,
            "expanded_route_id": self.expanded_route_id,
            "selection_reasons": list(self.selection_reasons),
            "max_wall_time_ms": self.max_wall_time_ms,
            "max_model_spend_micros": self.max_model_spend_micros,
            "max_expansion_token_budget": self.max_expansion_token_budget,
            "isolated_evaluation_worktree_required": (
                self.isolated_evaluation_worktree_required
            ),
            "expanded_is_oracle_candidate_only": self.expanded_is_oracle_candidate_only,
            "allow_external_expanded_disclosure": (
                self.allow_external_expanded_disclosure
            ),
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def plan_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_EXECUTION_PLAN_SCHEMA,
            "interface_id": SHADOW_EXECUTION_PLAN_INTERFACE,
            "header": self.header.to_dict(),
            "task_id": self.task_id,
            "audit_policy_cid": self.audit_policy_cid,
            "compressed_context_pack_cid": self.compressed_context_pack_cid,
            "expanded_context_pack_cid": self.expanded_context_pack_cid,
            "compressed_route_id": self.compressed_route_id,
            "expanded_route_id": self.expanded_route_id,
            "selection_reasons": list(self.selection_reasons),
            "max_wall_time_ms": self.max_wall_time_ms,
            "max_model_spend_micros": self.max_model_spend_micros,
            "max_expansion_token_budget": self.max_expansion_token_budget,
            "isolated_evaluation_worktree_required": (
                self.isolated_evaluation_worktree_required
            ),
            "expanded_is_oracle_candidate_only": self.expanded_is_oracle_candidate_only,
            "allow_external_expanded_disclosure": (
                self.allow_external_expanded_disclosure
            ),
            "metadata": _thaw_structured(self.metadata),
            "plan_cid": self.plan_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowExecutionPlan":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        schema = payload.pop("schema")
        interface_id = payload.pop("interface_id")
        if schema != SHADOW_EXECUTION_PLAN_SCHEMA:
            raise SemanticGovernorExecutionError(
                "unsupported ShadowExecutionPlan schema version"
            )
        if interface_id != SHADOW_EXECUTION_PLAN_INTERFACE:
            raise SemanticGovernorExecutionError(
                "unsupported ShadowExecutionPlan interface_id"
            )
        result = cls(
            header=payload["header"],
            task_id=payload["task_id"],
            audit_policy_cid=payload["audit_policy_cid"],
            compressed_context_pack_cid=payload["compressed_context_pack_cid"],
            expanded_context_pack_cid=payload["expanded_context_pack_cid"],
            compressed_route_id=payload["compressed_route_id"],
            expanded_route_id=payload["expanded_route_id"],
            selection_reasons=payload["selection_reasons"],
            max_wall_time_ms=payload["max_wall_time_ms"],
            max_model_spend_micros=payload["max_model_spend_micros"],
            max_expansion_token_budget=payload["max_expansion_token_budget"],
            isolated_evaluation_worktree_required=payload[
                "isolated_evaluation_worktree_required"
            ],
            expanded_is_oracle_candidate_only=payload[
                "expanded_is_oracle_candidate_only"
            ],
            allow_external_expanded_disclosure=payload[
                "allow_external_expanded_disclosure"
            ],
            metadata=payload["metadata"],
        )
        if claimed != result.plan_cid:
            raise SemanticGovernorExecutionError(
                "ShadowExecutionPlan plan_cid does not verify"
            )
        return result


# ---------------------------------------------------------------------------
# ShadowExecutionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowExecutionResult:
    """Result of executing a shadow plan's paired attempts.

    Expanded attempts remain candidate-only.  When disclosure policy forbids
    an external expanded call, ``expanded_attempt`` may be null with an
    explicit skip reason.
    """

    header: GovernorArtifactHeader
    plan_cid: str
    compressed_attempt: PairedAttemptRecord
    expanded_attempt: PairedAttemptRecord | None
    both_attempts_isolated: bool
    expanded_skipped_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "header",
            "plan_cid",
            "compressed_attempt",
            "expanded_attempt",
            "both_attempts_isolated",
            "expanded_skipped_reason",
            "metadata",
            "result_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "header",
            _normalize_header(self.header, expected_kind="shadow_execution_result"),
        )
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        if not isinstance(self.compressed_attempt, PairedAttemptRecord):
            raise SemanticGovernorExecutionError(
                "compressed_attempt must be PairedAttemptRecord"
            )
        if self.compressed_attempt.role != ShadowAttemptRole.COMPRESSED.value:
            raise SemanticGovernorExecutionError(
                "compressed_attempt.role must be compressed"
            )
        if self.expanded_attempt is not None:
            if not isinstance(self.expanded_attempt, PairedAttemptRecord):
                raise SemanticGovernorExecutionError(
                    "expanded_attempt must be PairedAttemptRecord or null"
                )
            if self.expanded_attempt.role != ShadowAttemptRole.EXPANDED.value:
                raise SemanticGovernorExecutionError(
                    "expanded_attempt.role must be expanded"
                )
            # Reinforce never-accepted invariant at the result boundary.
            assert_expanded_never_accepted(
                self.expanded_attempt.acceptance_disposition,
                role=self.expanded_attempt.role,
            )
            if self.expanded_skipped_reason is not None:
                raise SemanticGovernorExecutionError(
                    "expanded_skipped_reason must be null when expanded_attempt is present"
                )
        else:
            if not self.expanded_skipped_reason:
                raise SemanticGovernorExecutionError(
                    "expanded_skipped_reason is required when expanded_attempt is null"
                )
            object.__setattr__(
                self,
                "expanded_skipped_reason",
                _enum(
                    self.expanded_skipped_reason,
                    ShadowSelectionReason,
                    "expanded_skipped_reason",
                ),
            )
        object.__setattr__(
            self,
            "both_attempts_isolated",
            _bool(self.both_attempts_isolated, "both_attempts_isolated"),
        )
        if self.expanded_attempt is not None and not self.both_attempts_isolated:
            raise SemanticGovernorExecutionError(
                "paired expanded execution requires both_attempts_isolated"
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        # Header provenance execution_mode must not contradict attempts.
        header_mode = self.header.provenance.execution_mode
        if header_mode == ExecutionMode.SIMULATED.value:
            if self.compressed_attempt.execution_mode != ExecutionMode.SIMULATED.value:
                raise SemanticGovernorExecutionError(
                    "simulated header provenance requires simulated compressed_attempt"
                )
            if (
                self.expanded_attempt is not None
                and self.expanded_attempt.execution_mode
                != ExecutionMode.SIMULATED.value
            ):
                raise SemanticGovernorExecutionError(
                    "simulated header provenance requires simulated expanded_attempt"
                )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_EXECUTION_RESULT_SCHEMA,
            "interface_id": SHADOW_EXECUTION_RESULT_INTERFACE,
            "header": self.header.identity_payload(),
            "plan_cid": self.plan_cid,
            "compressed_attempt": self.compressed_attempt.identity_payload(),
            "expanded_attempt": (
                None
                if self.expanded_attempt is None
                else self.expanded_attempt.identity_payload()
            ),
            "both_attempts_isolated": self.both_attempts_isolated,
            "expanded_skipped_reason": self.expanded_skipped_reason,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_EXECUTION_RESULT_SCHEMA,
            "interface_id": SHADOW_EXECUTION_RESULT_INTERFACE,
            "header": self.header.to_dict(),
            "plan_cid": self.plan_cid,
            "compressed_attempt": self.compressed_attempt.to_dict(),
            "expanded_attempt": (
                None
                if self.expanded_attempt is None
                else self.expanded_attempt.to_dict()
            ),
            "both_attempts_isolated": self.both_attempts_isolated,
            "expanded_skipped_reason": self.expanded_skipped_reason,
            "metadata": _thaw_structured(self.metadata),
            "result_cid": self.result_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowExecutionResult":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("result_cid")
        schema = payload.pop("schema")
        interface_id = payload.pop("interface_id")
        if schema != SHADOW_EXECUTION_RESULT_SCHEMA:
            raise SemanticGovernorExecutionError(
                "unsupported ShadowExecutionResult schema version"
            )
        if interface_id != SHADOW_EXECUTION_RESULT_INTERFACE:
            raise SemanticGovernorExecutionError(
                "unsupported ShadowExecutionResult interface_id"
            )
        compressed_raw = payload["compressed_attempt"]
        if not isinstance(compressed_raw, Mapping):
            raise SemanticGovernorExecutionError(
                "compressed_attempt must be a mapping"
            )
        expanded_raw = payload["expanded_attempt"]
        expanded: PairedAttemptRecord | None
        if expanded_raw is None:
            expanded = None
        elif isinstance(expanded_raw, Mapping):
            expanded = PairedAttemptRecord.from_dict(expanded_raw)
        else:
            raise SemanticGovernorExecutionError(
                "expanded_attempt must be a mapping or null"
            )
        result = cls(
            header=payload["header"],
            plan_cid=payload["plan_cid"],
            compressed_attempt=PairedAttemptRecord.from_dict(compressed_raw),
            expanded_attempt=expanded,
            both_attempts_isolated=payload["both_attempts_isolated"],
            expanded_skipped_reason=payload["expanded_skipped_reason"],
            metadata=payload["metadata"],
        )
        if claimed != result.result_cid:
            raise SemanticGovernorExecutionError(
                "ShadowExecutionResult result_cid does not verify"
            )
        return result


# ---------------------------------------------------------------------------
# DifferentialPatchReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DifferentialPatchReport:
    """Semantic differential comparison beyond textual equality.

    Textual difference is an observation only.  Failure classification
    requires non-text bases (verification, AST/interface/effect evidence,
    human review, etc.).
    """

    header: GovernorArtifactHeader
    plan_cid: str
    shadow_result_cid: str
    text_differs: bool
    files_differ: bool
    symbols_differ: bool
    interfaces_differ: bool
    side_effects_differ: bool
    exceptions_differ: bool
    schemas_differ: bool
    tests_differ: bool
    proofs_differ: bool
    counterexamples_differ: bool
    static_analysis_differ: bool
    performance_differ: bool
    acceptance_differ: bool
    human_review_required: bool
    ast_edit_classes: Sequence[str]
    compressed_input_tokens: int
    expanded_input_tokens: int
    compressed_output_tokens: int
    expanded_output_tokens: int
    compressed_wall_time_ms: int
    expanded_wall_time_ms: int
    compressed_model_spend_micros: int
    expanded_model_spend_micros: int
    semantic_equivalent: bool | None
    failure_classified: bool
    classification_bases: Sequence[str]
    # Policy constant surface: textual difference is never semantic failure alone.
    textual_difference_is_not_semantic_failure: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "header",
            "plan_cid",
            "shadow_result_cid",
            "text_differs",
            "files_differ",
            "symbols_differ",
            "interfaces_differ",
            "side_effects_differ",
            "exceptions_differ",
            "schemas_differ",
            "tests_differ",
            "proofs_differ",
            "counterexamples_differ",
            "static_analysis_differ",
            "performance_differ",
            "acceptance_differ",
            "human_review_required",
            "ast_edit_classes",
            "compressed_input_tokens",
            "expanded_input_tokens",
            "compressed_output_tokens",
            "expanded_output_tokens",
            "compressed_wall_time_ms",
            "expanded_wall_time_ms",
            "compressed_model_spend_micros",
            "expanded_model_spend_micros",
            "semantic_equivalent",
            "failure_classified",
            "classification_bases",
            "textual_difference_is_not_semantic_failure",
            "metadata",
            "report_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "header",
            _normalize_header(self.header, expected_kind="differential_patch_report"),
        )
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self, "shadow_result_cid", _cid(self.shadow_result_cid, "shadow_result_cid")
        )
        for flag_name in (
            "text_differs",
            "files_differ",
            "symbols_differ",
            "interfaces_differ",
            "side_effects_differ",
            "exceptions_differ",
            "schemas_differ",
            "tests_differ",
            "proofs_differ",
            "counterexamples_differ",
            "static_analysis_differ",
            "performance_differ",
            "acceptance_differ",
            "human_review_required",
            "failure_classified",
            "textual_difference_is_not_semantic_failure",
        ):
            object.__setattr__(
                self, flag_name, _bool(getattr(self, flag_name), flag_name)
            )
        if not self.textual_difference_is_not_semantic_failure:
            raise SemanticGovernorExecutionError(
                "textual_difference_is_not_semantic_failure must be true; "
                "text difference alone cannot classify failure"
            )
        object.__setattr__(
            self,
            "ast_edit_classes",
            _unique_sorted_enums(
                list(self.ast_edit_classes),
                SemanticEditClass,
                "ast_edit_classes",
                max_items=MAX_EDIT_CLASSES,
            ),
        )
        for int_name in (
            "compressed_input_tokens",
            "expanded_input_tokens",
            "compressed_output_tokens",
            "expanded_output_tokens",
            "compressed_wall_time_ms",
            "expanded_wall_time_ms",
            "compressed_model_spend_micros",
            "expanded_model_spend_micros",
        ):
            object.__setattr__(
                self, int_name, _nonneg_int(getattr(self, int_name), int_name)
            )
        if self.semantic_equivalent is not None and type(self.semantic_equivalent) is not bool:
            raise SemanticGovernorExecutionError(
                "semantic_equivalent must be a boolean or null"
            )
        object.__setattr__(
            self,
            "classification_bases",
            _unique_sorted_enums(
                list(self.classification_bases),
                OutcomeClassificationBasis,
                "classification_bases",
                max_items=MAX_CLASSIFICATION_BASES,
            ),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        # Core acceptance criterion: text alone cannot classify failure.
        assert_failure_classification_not_text_alone(
            self.classification_bases,
            failure_classified=self.failure_classified,
        )
        if self.failure_classified and self.semantic_equivalent is True:
            raise SemanticGovernorExecutionError(
                "failure_classified is incompatible with semantic_equivalent=true"
            )
        # If only text differs among structural flags and no non-text bases,
        # failure cannot be asserted (covered above); also reject claiming
        # semantic_equivalent=false solely from text.
        only_text = self.text_differs and not any(
            (
                self.files_differ,
                self.symbols_differ,
                self.interfaces_differ,
                self.side_effects_differ,
                self.exceptions_differ,
                self.schemas_differ,
                self.tests_differ,
                self.proofs_differ,
                self.counterexamples_differ,
                self.static_analysis_differ,
                self.performance_differ,
                self.acceptance_differ,
            )
        )
        structural_edits = set(self.ast_edit_classes) - {
            SemanticEditClass.IDENTICAL.value,
            SemanticEditClass.EQUIVALENT_REFORMAT.value,
            SemanticEditClass.RENAME.value,
            SemanticEditClass.REORDER.value,
        }
        if only_text and not structural_edits:
            if self.failure_classified:
                raise SemanticGovernorExecutionError(
                    "text difference alone cannot classify failure"
                )
            if self.semantic_equivalent is False and not non_text_classification_bases(
                self.classification_bases
            ):
                raise SemanticGovernorExecutionError(
                    "semantic_equivalent=false requires non-text evidence; "
                    "text difference alone is not semantic failure"
                )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": DIFFERENTIAL_PATCH_REPORT_SCHEMA,
            "interface_id": DIFFERENTIAL_PATCH_REPORT_INTERFACE,
            "header": self.header.identity_payload(),
            "plan_cid": self.plan_cid,
            "shadow_result_cid": self.shadow_result_cid,
            "text_differs": self.text_differs,
            "files_differ": self.files_differ,
            "symbols_differ": self.symbols_differ,
            "interfaces_differ": self.interfaces_differ,
            "side_effects_differ": self.side_effects_differ,
            "exceptions_differ": self.exceptions_differ,
            "schemas_differ": self.schemas_differ,
            "tests_differ": self.tests_differ,
            "proofs_differ": self.proofs_differ,
            "counterexamples_differ": self.counterexamples_differ,
            "static_analysis_differ": self.static_analysis_differ,
            "performance_differ": self.performance_differ,
            "acceptance_differ": self.acceptance_differ,
            "human_review_required": self.human_review_required,
            "ast_edit_classes": list(self.ast_edit_classes),
            "compressed_input_tokens": self.compressed_input_tokens,
            "expanded_input_tokens": self.expanded_input_tokens,
            "compressed_output_tokens": self.compressed_output_tokens,
            "expanded_output_tokens": self.expanded_output_tokens,
            "compressed_wall_time_ms": self.compressed_wall_time_ms,
            "expanded_wall_time_ms": self.expanded_wall_time_ms,
            "compressed_model_spend_micros": self.compressed_model_spend_micros,
            "expanded_model_spend_micros": self.expanded_model_spend_micros,
            "semantic_equivalent": self.semantic_equivalent,
            "failure_classified": self.failure_classified,
            "classification_bases": list(self.classification_bases),
            "textual_difference_is_not_semantic_failure": (
                self.textual_difference_is_not_semantic_failure
            ),
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def report_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        # Seal nested header with full to_dict for durable export.
        value["header"] = self.header.to_dict()
        value["report_cid"] = self.report_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DifferentialPatchReport":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("report_cid")
        schema = payload.pop("schema")
        interface_id = payload.pop("interface_id")
        if schema != DIFFERENTIAL_PATCH_REPORT_SCHEMA:
            raise SemanticGovernorExecutionError(
                "unsupported DifferentialPatchReport schema version"
            )
        if interface_id != DIFFERENTIAL_PATCH_REPORT_INTERFACE:
            raise SemanticGovernorExecutionError(
                "unsupported DifferentialPatchReport interface_id"
            )
        result = cls(
            header=payload["header"],
            plan_cid=payload["plan_cid"],
            shadow_result_cid=payload["shadow_result_cid"],
            text_differs=payload["text_differs"],
            files_differ=payload["files_differ"],
            symbols_differ=payload["symbols_differ"],
            interfaces_differ=payload["interfaces_differ"],
            side_effects_differ=payload["side_effects_differ"],
            exceptions_differ=payload["exceptions_differ"],
            schemas_differ=payload["schemas_differ"],
            tests_differ=payload["tests_differ"],
            proofs_differ=payload["proofs_differ"],
            counterexamples_differ=payload["counterexamples_differ"],
            static_analysis_differ=payload["static_analysis_differ"],
            performance_differ=payload["performance_differ"],
            acceptance_differ=payload["acceptance_differ"],
            human_review_required=payload["human_review_required"],
            ast_edit_classes=payload["ast_edit_classes"],
            compressed_input_tokens=payload["compressed_input_tokens"],
            expanded_input_tokens=payload["expanded_input_tokens"],
            compressed_output_tokens=payload["compressed_output_tokens"],
            expanded_output_tokens=payload["expanded_output_tokens"],
            compressed_wall_time_ms=payload["compressed_wall_time_ms"],
            expanded_wall_time_ms=payload["expanded_wall_time_ms"],
            compressed_model_spend_micros=payload["compressed_model_spend_micros"],
            expanded_model_spend_micros=payload["expanded_model_spend_micros"],
            semantic_equivalent=payload["semantic_equivalent"],
            failure_classified=payload["failure_classified"],
            classification_bases=payload["classification_bases"],
            textual_difference_is_not_semantic_failure=payload[
                "textual_difference_is_not_semantic_failure"
            ],
            metadata=payload["metadata"],
        )
        if claimed != result.report_cid:
            raise SemanticGovernorExecutionError(
                "DifferentialPatchReport report_cid does not verify"
            )
        return result


# ---------------------------------------------------------------------------
# SemanticOutcomeComparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticOutcomeComparison:
    """Closed comparative outcome for a paired shadow evaluation."""

    header: GovernorArtifactHeader
    plan_cid: str
    shadow_result_cid: str
    differential_report_cid: str
    comparative_outcome: ComparativeOutcome | str
    compressed_acceptance: AcceptanceDisposition | str
    expanded_acceptance: AcceptanceDisposition | str
    human_review_required: bool
    classification_bases: Sequence[str]
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "header",
            "plan_cid",
            "shadow_result_cid",
            "differential_report_cid",
            "comparative_outcome",
            "compressed_acceptance",
            "expanded_acceptance",
            "human_review_required",
            "classification_bases",
            "notes",
            "metadata",
            "comparison_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "header",
            _normalize_header(self.header, expected_kind="semantic_outcome_comparison"),
        )
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self, "shadow_result_cid", _cid(self.shadow_result_cid, "shadow_result_cid")
        )
        object.__setattr__(
            self,
            "differential_report_cid",
            _cid(self.differential_report_cid, "differential_report_cid"),
        )
        outcome = _enum(
            self.comparative_outcome, ComparativeOutcome, "comparative_outcome"
        )
        object.__setattr__(self, "comparative_outcome", outcome)
        compressed = _enum(
            self.compressed_acceptance,
            AcceptanceDisposition,
            "compressed_acceptance",
        )
        object.__setattr__(self, "compressed_acceptance", compressed)
        expanded = _enum(
            self.expanded_acceptance,
            AcceptanceDisposition,
            "expanded_acceptance",
        )
        object.__setattr__(self, "expanded_acceptance", expanded)
        # Expanded acceptance never accepted by construction.
        assert_expanded_never_accepted(
            expanded, role=ShadowAttemptRole.EXPANDED.value, name="expanded_acceptance"
        )
        object.__setattr__(
            self,
            "human_review_required",
            _bool(self.human_review_required, "human_review_required"),
        )
        object.__setattr__(
            self,
            "classification_bases",
            _unique_sorted_enums(
                list(self.classification_bases),
                OutcomeClassificationBasis,
                "classification_bases",
                max_items=MAX_CLASSIFICATION_BASES,
            ),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        if outcome == ComparativeOutcome.HUMAN_REVIEW_REQUIRED.value:
            if not self.human_review_required:
                raise SemanticGovernorExecutionError(
                    "human_review_required outcome requires human_review_required=true"
                )
            if compressed not in {
                AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value,
                AcceptanceDisposition.NOT_ACCEPTED.value,
            }:
                raise SemanticGovernorExecutionError(
                    "human_review_required outcome forbids compressed acceptance"
                )

        # Failure-like comparative outcomes require non-text classification bases.
        failure_like = outcome in _FAILURE_LIKE_OUTCOMES
        assert_failure_classification_not_text_alone(
            self.classification_bases,
            failure_classified=failure_like,
            name="classification_bases",
        )

        if (
            outcome == ComparativeOutcome.EQUIVALENT_SUCCESS.value
            and compressed == AcceptanceDisposition.ACCEPTED.value
            and self.header.provenance.execution_mode
            == ExecutionMode.SIMULATED.value
        ):
            raise SemanticGovernorExecutionError(
                "simulated provenance cannot accept compressed output"
            )

        if expanded == AcceptanceDisposition.ACCEPTED.value:
            # Defensive; assert_expanded_never_accepted already covers this.
            raise SemanticGovernorExecutionError(
                "expanded_acceptance is never accepted by construction"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_OUTCOME_COMPARISON_SCHEMA,
            "interface_id": SEMANTIC_OUTCOME_COMPARISON_INTERFACE,
            "header": self.header.identity_payload(),
            "plan_cid": self.plan_cid,
            "shadow_result_cid": self.shadow_result_cid,
            "differential_report_cid": self.differential_report_cid,
            "comparative_outcome": self.comparative_outcome,
            "compressed_acceptance": self.compressed_acceptance,
            "expanded_acceptance": self.expanded_acceptance,
            "human_review_required": self.human_review_required,
            "classification_bases": list(self.classification_bases),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def comparison_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["header"] = self.header.to_dict()
        value["comparison_cid"] = self.comparison_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticOutcomeComparison":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("comparison_cid")
        schema = payload.pop("schema")
        interface_id = payload.pop("interface_id")
        if schema != SEMANTIC_OUTCOME_COMPARISON_SCHEMA:
            raise SemanticGovernorExecutionError(
                "unsupported SemanticOutcomeComparison schema version"
            )
        if interface_id != SEMANTIC_OUTCOME_COMPARISON_INTERFACE:
            raise SemanticGovernorExecutionError(
                "unsupported SemanticOutcomeComparison interface_id"
            )
        result = cls(
            header=payload["header"],
            plan_cid=payload["plan_cid"],
            shadow_result_cid=payload["shadow_result_cid"],
            differential_report_cid=payload["differential_report_cid"],
            comparative_outcome=payload["comparative_outcome"],
            compressed_acceptance=payload["compressed_acceptance"],
            expanded_acceptance=payload["expanded_acceptance"],
            human_review_required=payload["human_review_required"],
            classification_bases=payload["classification_bases"],
            notes=payload["notes"],
            metadata=payload["metadata"],
        )
        if claimed != result.comparison_cid:
            raise SemanticGovernorExecutionError(
                "SemanticOutcomeComparison comparison_cid does not verify"
            )
        return result


def verify_plan_identity(plan: ShadowExecutionPlan | Mapping[str, Any]) -> str:
    """Recompute and return plan_cid; raise on forged or malformed input."""

    sealed = plan if isinstance(plan, ShadowExecutionPlan) else ShadowExecutionPlan.from_dict(plan)
    recomputed = cid_for_structured(sealed.identity_payload())
    if recomputed != sealed.plan_cid:
        raise SemanticGovernorExecutionError(
            "plan_cid does not match recomputed identity"
        )
    return recomputed


def verify_result_identity(result: ShadowExecutionResult | Mapping[str, Any]) -> str:
    """Recompute and return result_cid; raise on forged or malformed input."""

    sealed = (
        result
        if isinstance(result, ShadowExecutionResult)
        else ShadowExecutionResult.from_dict(result)
    )
    recomputed = cid_for_structured(sealed.identity_payload())
    if recomputed != sealed.result_cid:
        raise SemanticGovernorExecutionError(
            "result_cid does not match recomputed identity"
        )
    return recomputed


def verify_report_identity(report: DifferentialPatchReport | Mapping[str, Any]) -> str:
    """Recompute and return report_cid; raise on forged or malformed input."""

    sealed = (
        report
        if isinstance(report, DifferentialPatchReport)
        else DifferentialPatchReport.from_dict(report)
    )
    recomputed = cid_for_structured(sealed.identity_payload())
    if recomputed != sealed.report_cid:
        raise SemanticGovernorExecutionError(
            "report_cid does not match recomputed identity"
        )
    return recomputed


def verify_comparison_identity(
    comparison: SemanticOutcomeComparison | Mapping[str, Any],
) -> str:
    """Recompute and return comparison_cid; raise on forged or malformed input."""

    sealed = (
        comparison
        if isinstance(comparison, SemanticOutcomeComparison)
        else SemanticOutcomeComparison.from_dict(comparison)
    )
    recomputed = cid_for_structured(sealed.identity_payload())
    if recomputed != sealed.comparison_cid:
        raise SemanticGovernorExecutionError(
            "comparison_cid does not match recomputed identity"
        )
    return recomputed


__all__ = [
    "AcceptanceDisposition",
    "AttemptTerminalStatus",
    "COST_TIMING_SCHEMA",
    "ComparativeOutcome",
    "CostTimingProjection",
    "DIFFERENTIAL_PATCH_REPORT_INTERFACE",
    "DIFFERENTIAL_PATCH_REPORT_SCHEMA",
    "DifferentialPatchReport",
    "OutcomeClassificationBasis",
    "PAIRED_ATTEMPT_SCHEMA",
    "PairedAttemptRecord",
    "SCG_EXECUTION_CONTRACTS_EVIDENCE",
    "SEMANTIC_GOVERNOR_EXECUTION_INTERFACE",
    "SEMANTIC_OUTCOME_COMPARISON_INTERFACE",
    "SEMANTIC_OUTCOME_COMPARISON_SCHEMA",
    "SHADOW_EXECUTION_PLAN_INTERFACE",
    "SHADOW_EXECUTION_PLAN_SCHEMA",
    "SHADOW_EXECUTION_RESULT_INTERFACE",
    "SHADOW_EXECUTION_RESULT_SCHEMA",
    "SemanticEditClass",
    "SemanticGovernorExecutionError",
    "SemanticOutcomeComparison",
    "ShadowAttemptRole",
    "ShadowExecutionPlan",
    "ShadowExecutionResult",
    "ShadowSelectionReason",
    "VERIFICATION_PROJECTION_SCHEMA",
    "VerificationProjection",
    "assert_expanded_never_accepted",
    "assert_failure_classification_not_text_alone",
    "comparative_outcomes",
    "non_text_classification_bases",
    "outcome_classification_bases",
    "verify_comparison_identity",
    "verify_plan_identity",
    "verify_report_identity",
    "verify_result_identity",
    # Re-exported base symbols commonly needed by consumers/tests.
    "ArtifactProvenance",
    "AuthoritySource",
    "ExecutionMode",
    "GeneratorIdentity",
    "GovernorArtifactHeader",
    "GovernorAssumption",
    "GovernorTerminalStatus",
]
