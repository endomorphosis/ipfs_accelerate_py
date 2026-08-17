"""Compare semantic patch outcomes beyond text (SCG-027).

``compare_shadow_results`` projects paired compressed/expanded shadow attempts
into a :class:`DifferentialPatchReport` and closed
:class:`SemanticOutcomeComparison`.

Normative fail-closed invariants:

* Textual difference is an observation only — it never classifies failure alone.
* Semantic equivalence uses structural and verification evidence; model
  agreement and textual equality are not proof of equivalence.
* ``compressed_failed_expanded_succeeded`` is a distinct comparative outcome.
* Inconclusive verification (or non-terminal attempt status) stays
  ``verification_inconclusive``; it is never upgraded to success or failure.
* Expanded acceptance remains oracle/candidate only (never ``accepted``).

Conflict policy: reuses ``DifferentialPatchReport`` / ``SemanticOutcomeComparison``
contracts and attempt verification projections; does not mint a second receipt
hierarchy or treat text equality as semantic success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Iterable, Mapping, Sequence
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
    validate_structured_value,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AssumptionKind,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
    SemanticGovernorBaseError,
    reject_private_and_model_authority,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    AttemptTerminalStatus,
    ComparativeOutcome,
    DifferentialPatchReport,
    OutcomeClassificationBasis,
    PairedAttemptRecord,
    SemanticEditClass,
    SemanticGovernorExecutionError,
    SemanticOutcomeComparison,
    ShadowAttemptRole,
    ShadowExecutionResult,
    VerificationProjection,
    assert_expanded_never_accepted,
    assert_failure_classification_not_text_alone,
    non_text_classification_bases,
    verify_comparison_identity,
    verify_report_identity,
    verify_result_identity,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_DIFFERENTIAL_EVIDENCE: Final[str] = "scg/differential@1"
COMPARE_SHADOW_RESULTS_INTERFACE: Final[str] = "compare_shadow_results@1"
SEMANTIC_DIFFERENTIAL_OUTCOME_INTERFACE: Final[str] = "SemanticDifferentialOutcome@1"
SEMANTIC_DIFFERENTIAL_OUTCOME_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "semantic-differential-outcome@1"
)
STRUCTURAL_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "attempt-structural-projection@1"
)
STRUCTURAL_COMPARISON_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "structural-comparison-evidence@1"
)

GENERATOR_ID: Final[str] = "semantic_governor_differential"
GENERATOR_VERSION: Final[str] = "1.0.0"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_IDS: Final[int] = 256
MAX_EDIT_CLASSES: Final[int] = 256
MAX_METADATA_KEYS: Final[int] = 64

# AST edit classes that alone do not prove semantic divergence.
_EQUIVALENCE_PRESERVING_EDITS: Final[frozenset[str]] = frozenset(
    {
        SemanticEditClass.IDENTICAL.value,
        SemanticEditClass.EQUIVALENT_REFORMAT.value,
        SemanticEditClass.RENAME.value,
        SemanticEditClass.REORDER.value,
    }
)

# Edit classes that prove structural non-equivalence when present.
_DIVERGENT_EDITS: Final[frozenset[str]] = frozenset(
    {
        SemanticEditClass.ADD.value,
        SemanticEditClass.REMOVE.value,
        SemanticEditClass.MODIFY_LOGIC.value,
        SemanticEditClass.INTERFACE_CHANGE.value,
    }
)

_SUCCESS_STATUSES: Final[frozenset[str]] = frozenset(
    {AttemptTerminalStatus.SUCCEEDED.value}
)
_FAILURE_STATUSES: Final[frozenset[str]] = frozenset(
    {AttemptTerminalStatus.FAILED.value}
)
_INCONCLUSIVE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        AttemptTerminalStatus.INCONCLUSIVE.value,
        AttemptTerminalStatus.CANCELLED.value,
        AttemptTerminalStatus.EVALUATION_FAILED.value,
        AttemptTerminalStatus.SKIPPED.value,
    }
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


class SemanticGovernorDifferentialError(SemanticGovernorExecutionError):
    """Raised when differential comparison input is malformed or fail-closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "differential_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


# ---------------------------------------------------------------------------
# Validation helpers (local, closed)
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise SemanticGovernorDifferentialError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise SemanticGovernorDifferentialError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise SemanticGovernorDifferentialError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise SemanticGovernorDifferentialError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SemanticGovernorDifferentialError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise SemanticGovernorDifferentialError(
            f"{name} must be a nonnegative integer"
        )
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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticGovernorDifferentialError(f"{name} must be a mapping")
    if len(value) > MAX_METADATA_KEYS:
        raise SemanticGovernorDifferentialError(f"{name} exceeds metadata key bound")
    thawed = _thaw_structured(dict(value))
    try:
        validate_structured_value(thawed, path=name)
    except Exception as exc:
        raise SemanticGovernorDifferentialError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    try:
        reject_private_and_model_authority(thawed, path=name)
    except SemanticGovernorBaseError as exc:
        raise SemanticGovernorDifferentialError(str(exc)) from exc
    return _freeze_structured(thawed)


def _unique_sorted_tokens(
    values: Iterable[Any], name: str, *, max_items: int = MAX_IDS
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SemanticGovernorDifferentialError(f"{name} must be a list or tuple")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        token = _text(item, name)
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    ordered.sort()
    if len(ordered) > max_items:
        raise SemanticGovernorDifferentialError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _unique_sorted_edit_classes(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SemanticGovernorDifferentialError(f"{name} must be a list or tuple")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        try:
            edit = SemanticEditClass(item).value
        except (TypeError, ValueError) as exc:
            raise SemanticGovernorDifferentialError(
                f"{name} has unsupported edit class {item!r}"
            ) from exc
        if edit in seen:
            continue
        seen.add(edit)
        ordered.append(edit)
    ordered.sort()
    if len(ordered) > MAX_EDIT_CLASSES:
        raise SemanticGovernorDifferentialError(f"{name} exceeds maximum length")
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Structural projections (optional inputs for beyond-text comparison)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttemptStructuralProjection:
    """Closed structural projection of one attempt's patch for differential use.

    All identity sets are managed references (paths, symbol ids, digests) —
    never raw private source.  Empty projections are admitted and mean
    "no structural evidence supplied" rather than "identical empty patch".
    """

    text_digest: str | None = None
    file_ids: Sequence[str] = ()
    symbol_ids: Sequence[str] = ()
    interface_ids: Sequence[str] = ()
    side_effect_ids: Sequence[str] = ()
    exception_contracts: Sequence[str] = ()
    schema_ids: Sequence[str] = ()
    ast_edit_classes: Sequence[str] = ()
    performance_digest: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "text_digest", _optional_text(self.text_digest, "text_digest")
        )
        object.__setattr__(
            self,
            "file_ids",
            _unique_sorted_tokens(list(self.file_ids), "file_ids"),
        )
        object.__setattr__(
            self,
            "symbol_ids",
            _unique_sorted_tokens(list(self.symbol_ids), "symbol_ids"),
        )
        object.__setattr__(
            self,
            "interface_ids",
            _unique_sorted_tokens(list(self.interface_ids), "interface_ids"),
        )
        object.__setattr__(
            self,
            "side_effect_ids",
            _unique_sorted_tokens(list(self.side_effect_ids), "side_effect_ids"),
        )
        object.__setattr__(
            self,
            "exception_contracts",
            _unique_sorted_tokens(
                list(self.exception_contracts), "exception_contracts"
            ),
        )
        object.__setattr__(
            self,
            "schema_ids",
            _unique_sorted_tokens(list(self.schema_ids), "schema_ids"),
        )
        object.__setattr__(
            self,
            "ast_edit_classes",
            _unique_sorted_edit_classes(
                list(self.ast_edit_classes), "ast_edit_classes"
            ),
        )
        object.__setattr__(
            self,
            "performance_digest",
            _optional_text(self.performance_digest, "performance_digest"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def is_empty(self) -> bool:
        return (
            self.text_digest is None
            and not self.file_ids
            and not self.symbol_ids
            and not self.interface_ids
            and not self.side_effect_ids
            and not self.exception_contracts
            and not self.schema_ids
            and not self.ast_edit_classes
            and self.performance_digest is None
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": STRUCTURAL_PROJECTION_SCHEMA,
            "text_digest": self.text_digest,
            "file_ids": list(self.file_ids),
            "symbol_ids": list(self.symbol_ids),
            "interface_ids": list(self.interface_ids),
            "side_effect_ids": list(self.side_effect_ids),
            "exception_contracts": list(self.exception_contracts),
            "schema_ids": list(self.schema_ids),
            "ast_edit_classes": list(self.ast_edit_classes),
            "performance_digest": self.performance_digest,
            "metadata": _thaw_structured(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttemptStructuralProjection":
        if not isinstance(data, Mapping):
            raise SemanticGovernorDifferentialError(
                "AttemptStructuralProjection must be a mapping"
            )
        schema = data.get("schema")
        if schema is not None and schema != STRUCTURAL_PROJECTION_SCHEMA:
            raise SemanticGovernorDifferentialError(
                "unsupported AttemptStructuralProjection schema version"
            )
        return cls(
            text_digest=data.get("text_digest"),
            file_ids=data.get("file_ids") or (),
            symbol_ids=data.get("symbol_ids") or (),
            interface_ids=data.get("interface_ids") or (),
            side_effect_ids=data.get("side_effect_ids") or (),
            exception_contracts=data.get("exception_contracts") or (),
            schema_ids=data.get("schema_ids") or (),
            ast_edit_classes=data.get("ast_edit_classes") or (),
            performance_digest=data.get("performance_digest"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True, slots=True)
class StructuralComparisonEvidence:
    """Paired structural projections plus optional pairwise edit classes."""

    compressed: AttemptStructuralProjection
    expanded: AttemptStructuralProjection
    pairwise_ast_edit_classes: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.compressed, AttemptStructuralProjection):
            raise SemanticGovernorDifferentialError(
                "compressed must be AttemptStructuralProjection"
            )
        if not isinstance(self.expanded, AttemptStructuralProjection):
            raise SemanticGovernorDifferentialError(
                "expanded must be AttemptStructuralProjection"
            )
        object.__setattr__(
            self,
            "pairwise_ast_edit_classes",
            _unique_sorted_edit_classes(
                list(self.pairwise_ast_edit_classes), "pairwise_ast_edit_classes"
            ),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": STRUCTURAL_COMPARISON_EVIDENCE_SCHEMA,
            "compressed": self.compressed.identity_payload(),
            "expanded": self.expanded.identity_payload(),
            "pairwise_ast_edit_classes": list(self.pairwise_ast_edit_classes),
            "metadata": _thaw_structured(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuralComparisonEvidence":
        if not isinstance(data, Mapping):
            raise SemanticGovernorDifferentialError(
                "StructuralComparisonEvidence must be a mapping"
            )
        schema = data.get("schema")
        if schema is not None and schema != STRUCTURAL_COMPARISON_EVIDENCE_SCHEMA:
            raise SemanticGovernorDifferentialError(
                "unsupported StructuralComparisonEvidence schema version"
            )
        compressed_raw = data.get("compressed")
        expanded_raw = data.get("expanded")
        if not isinstance(compressed_raw, Mapping) or not isinstance(
            expanded_raw, Mapping
        ):
            raise SemanticGovernorDifferentialError(
                "compressed and expanded structural projections must be mappings"
            )
        return cls(
            compressed=AttemptStructuralProjection.from_dict(compressed_raw),
            expanded=AttemptStructuralProjection.from_dict(expanded_raw),
            pairwise_ast_edit_classes=data.get("pairwise_ast_edit_classes") or (),
            metadata=data.get("metadata") or {},
        )


# ---------------------------------------------------------------------------
# Outcome envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticDifferentialOutcome:
    """Sealed differential report + comparative outcome for one paired audit."""

    report: DifferentialPatchReport
    comparison: SemanticOutcomeComparison
    metadata: Mapping[str, Any] = field(default_factory=dict)

    INTERFACE: Final[str] = SEMANTIC_DIFFERENTIAL_OUTCOME_INTERFACE
    SCHEMA: Final[str] = SEMANTIC_DIFFERENTIAL_OUTCOME_SCHEMA
    EVIDENCE: Final[str] = SCG_DIFFERENTIAL_EVIDENCE

    def __post_init__(self) -> None:
        if not isinstance(self.report, DifferentialPatchReport):
            raise SemanticGovernorDifferentialError(
                "report must be DifferentialPatchReport"
            )
        if not isinstance(self.comparison, SemanticOutcomeComparison):
            raise SemanticGovernorDifferentialError(
                "comparison must be SemanticOutcomeComparison"
            )
        if self.comparison.differential_report_cid != self.report.report_cid:
            raise SemanticGovernorDifferentialError(
                "comparison.differential_report_cid must equal report.report_cid"
            )
        if self.comparison.shadow_result_cid != self.report.shadow_result_cid:
            raise SemanticGovernorDifferentialError(
                "comparison and report shadow_result_cid must match"
            )
        if self.comparison.plan_cid != self.report.plan_cid:
            raise SemanticGovernorDifferentialError(
                "comparison and report plan_cid must match"
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def comparative_outcome(self) -> str:
        return str(self.comparison.comparative_outcome)

    @property
    def semantic_equivalent(self) -> bool | None:
        return self.report.semantic_equivalent

    @property
    def failure_classified(self) -> bool:
        return bool(self.report.failure_classified)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_DIFFERENTIAL_OUTCOME_SCHEMA,
            "interface_id": SEMANTIC_DIFFERENTIAL_OUTCOME_INTERFACE,
            "evidence": SCG_DIFFERENTIAL_EVIDENCE,
            "report": self.report.identity_payload(),
            "comparison": self.comparison.identity_payload(),
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def outcome_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_DIFFERENTIAL_OUTCOME_SCHEMA,
            "interface_id": SEMANTIC_DIFFERENTIAL_OUTCOME_INTERFACE,
            "evidence": SCG_DIFFERENTIAL_EVIDENCE,
            "report": self.report.to_dict(),
            "comparison": self.comparison.to_dict(),
            "metadata": _thaw_structured(self.metadata),
            "outcome_cid": self.outcome_cid,
            "comparative_outcome": self.comparative_outcome,
        }


# ---------------------------------------------------------------------------
# Attempt / verification normalization
# ---------------------------------------------------------------------------


def _require_attempt(
    value: Any, *, expected_role: str | None = None, name: str = "attempt"
) -> PairedAttemptRecord:
    if isinstance(value, PairedAttemptRecord):
        attempt = value
    elif isinstance(value, Mapping):
        try:
            attempt = PairedAttemptRecord.from_dict(value)
        except SemanticGovernorExecutionError as exc:
            raise SemanticGovernorDifferentialError(
                f"invalid {name}: {exc}",
                reason_code="invalid_attempt",
            ) from exc
    else:
        raise SemanticGovernorDifferentialError(
            f"{name} must be PairedAttemptRecord or mapping",
            reason_code="invalid_attempt",
        )
    if expected_role is not None and attempt.role != expected_role:
        raise SemanticGovernorDifferentialError(
            f"{name}.role must be {expected_role!r}, got {attempt.role!r}",
            reason_code="role_mismatch",
        )
    return attempt


def _require_shadow_result(value: Any) -> ShadowExecutionResult:
    if isinstance(value, ShadowExecutionResult):
        result = value
    elif isinstance(value, Mapping):
        try:
            result = ShadowExecutionResult.from_dict(value)
        except SemanticGovernorExecutionError as exc:
            raise SemanticGovernorDifferentialError(
                f"invalid shadow result: {exc}",
                reason_code="invalid_shadow_result",
            ) from exc
    else:
        raise SemanticGovernorDifferentialError(
            "shadow_result must be ShadowExecutionResult or mapping",
            reason_code="invalid_shadow_result",
        )
    try:
        verify_result_identity(result)
    except SemanticGovernorExecutionError as exc:
        raise SemanticGovernorDifferentialError(
            f"shadow result identity failed: {exc}",
            reason_code="invalid_shadow_result",
        ) from exc
    return result


def _require_structural(
    value: Any | None,
) -> StructuralComparisonEvidence | None:
    if value is None:
        return None
    if isinstance(value, StructuralComparisonEvidence):
        return value
    if isinstance(value, Mapping):
        return StructuralComparisonEvidence.from_dict(value)
    raise SemanticGovernorDifferentialError(
        "structural_evidence must be StructuralComparisonEvidence or mapping",
        reason_code="invalid_structural_evidence",
    )


def _verification_from_evidence(
    evidence: Any | None,
    *,
    fallback: VerificationProjection,
) -> VerificationProjection:
    """Project optional verification evidence onto a closed projection.

    Accepts ``VerificationProjection``, a mapping of projection fields, or an
    audit-evidence-like object exposing a ``verification`` attribute.  Never
    upgrades statuses; missing evidence keeps the attempt fallback.
    """

    if evidence is None:
        return fallback
    if isinstance(evidence, VerificationProjection):
        return evidence
    if hasattr(evidence, "verification"):
        nested = getattr(evidence, "verification")
        if isinstance(nested, VerificationProjection):
            return nested
        if isinstance(nested, Mapping):
            try:
                return VerificationProjection.from_dict(nested)
            except SemanticGovernorExecutionError as exc:
                raise SemanticGovernorDifferentialError(
                    f"invalid nested verification evidence: {exc}",
                    reason_code="invalid_verification_evidence",
                ) from exc
    if isinstance(evidence, Mapping):
        if "verification" in evidence and isinstance(evidence["verification"], Mapping):
            try:
                return VerificationProjection.from_dict(evidence["verification"])
            except SemanticGovernorExecutionError as exc:
                raise SemanticGovernorDifferentialError(
                    f"invalid verification evidence: {exc}",
                    reason_code="invalid_verification_evidence",
                ) from exc
        # Projection-shaped mapping.
        if "verification_bundle_cid" in evidence:
            try:
                return VerificationProjection.from_dict(evidence)
            except SemanticGovernorExecutionError as exc:
                raise SemanticGovernorDifferentialError(
                    f"invalid verification projection: {exc}",
                    reason_code="invalid_verification_evidence",
                ) from exc
    raise SemanticGovernorDifferentialError(
        "verification_evidence must project to VerificationProjection",
        reason_code="invalid_verification_evidence",
    )


def _tri_bool_diff(left: bool | None, right: bool | None) -> bool | None:
    """Return whether two optional booleans differ.

    ``None`` on either side means the check was not evaluated — difference is
    inconclusive (returns ``None``) rather than a definitive diverge.
    """

    if left is None or right is None:
        return None
    return left != right


def _attempt_bucket(attempt: PairedAttemptRecord) -> str:
    """Map attempt terminal status to success / failure / inconclusive."""

    status = str(attempt.attempt_status)
    if status in _SUCCESS_STATUSES:
        return "success"
    if status in _FAILURE_STATUSES:
        return "failure"
    if status in _INCONCLUSIVE_STATUSES:
        return "inconclusive"
    return "inconclusive"


def _verification_is_inconclusive(verification: VerificationProjection) -> bool:
    """True when verification cannot support a decisive comparative outcome.

    A fully-evaluated failure (explicit false flags / counterexample) is
    decisive.  Missing matrix satisfaction with no explicit pass/fail leaves
    the audit inconclusive.
    """

    # Explicit decisive failure evidence is not "inconclusive".
    if verification.counterexample_present:
        return False
    evaluated = (
        verification.selected_tests_passed,
        verification.full_suite_passed,
        verification.proofs_passed,
        verification.static_checks_passed,
    )
    if all(flag is None for flag in evaluated) and not verification.acceptance_matrix_satisfied:
        return True
    # Mixed None with no acceptance matrix and no production eligibility:
    # treat as inconclusive only when every present flag is True-or-None and
    # acceptance is unsatisfied (cannot classify production success).
    if not verification.acceptance_matrix_satisfied and not verification.production_eligible:
        # If any evaluated flag is explicitly False, verification is decisive fail.
        if any(flag is False for flag in evaluated):
            return False
        # If all non-None flags are True but matrix still unsatisfied, evidence
        # is incomplete → inconclusive for comparative classification.
        if any(flag is None for flag in evaluated):
            return True
    return False


def _score_verification(verification: VerificationProjection) -> int:
    """Monotonic integer score for relative quality (higher is better)."""

    score = 0
    for flag in (
        verification.selected_tests_passed,
        verification.full_suite_passed,
        verification.proofs_passed,
        verification.static_checks_passed,
    ):
        if flag is True:
            score += 2
        elif flag is False:
            score -= 2
    if verification.acceptance_matrix_satisfied:
        score += 3
    if verification.production_eligible:
        score += 1
    if verification.counterexample_present:
        score -= 4
    return score


# ---------------------------------------------------------------------------
# Diff projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DiffFlags:
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
    ast_edit_classes: tuple[str, ...]
    semantic_equivalent: bool | None
    classification_bases: tuple[str, ...]
    structural_evidence_present: bool


def _set_differs(left: Sequence[str], right: Sequence[str]) -> bool:
    return frozenset(left) != frozenset(right)


def _project_diff_flags(
    *,
    compressed: PairedAttemptRecord,
    expanded: PairedAttemptRecord | None,
    compressed_verification: VerificationProjection,
    expanded_verification: VerificationProjection | None,
    structural: StructuralComparisonEvidence | None,
    force_human_review: bool,
) -> _DiffFlags:
    bases: set[str] = set()

    if expanded is None or expanded_verification is None:
        # Missing expanded oracle — no structural comparison possible.
        return _DiffFlags(
            text_differs=False,
            files_differ=False,
            symbols_differ=False,
            interfaces_differ=False,
            side_effects_differ=False,
            exceptions_differ=False,
            schemas_differ=False,
            tests_differ=False,
            proofs_differ=False,
            counterexamples_differ=False,
            static_analysis_differ=False,
            performance_differ=False,
            acceptance_differ=False,
            human_review_required=True,
            ast_edit_classes=(SemanticEditClass.UNKNOWN.value,),
            semantic_equivalent=None,
            classification_bases=(
                OutcomeClassificationBasis.HUMAN_REVIEW.value,
            ),
            structural_evidence_present=False,
        )

    c_cost = compressed.cost_timing
    e_cost = expanded.cost_timing

    # --- verification-derived flags ---
    tests_tri = _tri_bool_diff(
        compressed_verification.selected_tests_passed,
        expanded_verification.selected_tests_passed,
    )
    full_tri = _tri_bool_diff(
        compressed_verification.full_suite_passed,
        expanded_verification.full_suite_passed,
    )
    tests_differ = bool(tests_tri is True or full_tri is True)
    if tests_differ:
        bases.add(OutcomeClassificationBasis.TEST_RESULT_DIFF.value)
        bases.add(OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value)

    proofs_tri = _tri_bool_diff(
        compressed_verification.proofs_passed, expanded_verification.proofs_passed
    )
    proofs_differ = bool(proofs_tri is True)
    if proofs_differ:
        bases.add(OutcomeClassificationBasis.PROOF_RECEIPTS.value)
        bases.add(OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value)

    static_tri = _tri_bool_diff(
        compressed_verification.static_checks_passed,
        expanded_verification.static_checks_passed,
    )
    static_analysis_differ = bool(static_tri is True)
    if static_analysis_differ:
        bases.add(OutcomeClassificationBasis.STATIC_ANALYSIS_DIFF.value)

    counterexamples_differ = (
        compressed_verification.counterexample_present
        != expanded_verification.counterexample_present
    )
    if counterexamples_differ:
        bases.add(OutcomeClassificationBasis.COUNTEREXAMPLE_RECEIPTS.value)

    # Expanded is oracle/candidate-only by construction, so disposition
    # asymmetry (e.g. not_accepted vs candidate_only) is not semantic
    # divergence.  Compare the acceptance *matrix* and production
    # eligibility projections instead.
    acceptance_differ = (
        compressed_verification.acceptance_matrix_satisfied
        != expanded_verification.acceptance_matrix_satisfied
    )
    if acceptance_differ:
        bases.add(OutcomeClassificationBasis.ACCEPTANCE_MATRIX_DIFF.value)

    # Cost / timing observation (never failure alone).
    cost_differs = (
        c_cost.input_tokens != e_cost.input_tokens
        or c_cost.output_tokens != e_cost.output_tokens
        or c_cost.wall_time_ms != e_cost.wall_time_ms
        or c_cost.model_spend_micros != e_cost.model_spend_micros
    )
    if cost_differs:
        bases.add(OutcomeClassificationBasis.COST_TIMING.value)

    # Patch CID identity is an observation about produced artifacts, not proof
    # of semantic nonequivalence by itself.
    text_differs = (
        compressed.patch_cid is not None
        and expanded.patch_cid is not None
        and compressed.patch_cid != expanded.patch_cid
    )

    files_differ = False
    symbols_differ = False
    interfaces_differ = False
    side_effects_differ = False
    exceptions_differ = False
    schemas_differ = False
    performance_differ = cost_differs  # default from timings; may refine below
    ast_edit_classes: tuple[str, ...] = ()
    structural_present = False

    if structural is not None:
        c_struct = structural.compressed
        e_struct = structural.expanded
        structural_present = not (c_struct.is_empty() and e_struct.is_empty())

        if c_struct.text_digest is not None and e_struct.text_digest is not None:
            text_differs = c_struct.text_digest != e_struct.text_digest
        elif c_struct.text_digest is not None or e_struct.text_digest is not None:
            text_differs = True

        if text_differs:
            bases.add(OutcomeClassificationBasis.TEXT_DIFF.value)

        if not c_struct.is_empty() or not e_struct.is_empty():
            files_differ = _set_differs(c_struct.file_ids, e_struct.file_ids)
            symbols_differ = _set_differs(c_struct.symbol_ids, e_struct.symbol_ids)
            interfaces_differ = _set_differs(
                c_struct.interface_ids, e_struct.interface_ids
            )
            side_effects_differ = _set_differs(
                c_struct.side_effect_ids, e_struct.side_effect_ids
            )
            exceptions_differ = _set_differs(
                c_struct.exception_contracts, e_struct.exception_contracts
            )
            schemas_differ = _set_differs(c_struct.schema_ids, e_struct.schema_ids)

            if files_differ or symbols_differ:
                bases.add(OutcomeClassificationBasis.AST_EDIT_CLASSES.value)
            if interfaces_differ:
                bases.add(OutcomeClassificationBasis.INTERFACE_DIFF.value)
            if side_effects_differ:
                bases.add(OutcomeClassificationBasis.SIDE_EFFECT_DIFF.value)
            if exceptions_differ:
                bases.add(OutcomeClassificationBasis.EXCEPTION_CONTRACT_DIFF.value)
            if schemas_differ:
                bases.add(OutcomeClassificationBasis.SCHEMA_DIFF.value)

            if (
                c_struct.performance_digest is not None
                and e_struct.performance_digest is not None
            ):
                performance_differ = (
                    c_struct.performance_digest != e_struct.performance_digest
                )
            if performance_differ:
                bases.add(OutcomeClassificationBasis.PERFORMANCE_DIFF.value)

            # Pairwise AST classes take precedence when supplied.
            if structural.pairwise_ast_edit_classes:
                ast_edit_classes = tuple(structural.pairwise_ast_edit_classes)
            else:
                # Union of per-side classes projected against a common base.
                combined = sorted(
                    set(c_struct.ast_edit_classes) | set(e_struct.ast_edit_classes)
                )
                if not combined:
                    if (
                        not files_differ
                        and not symbols_differ
                        and not interfaces_differ
                        and not side_effects_differ
                        and not exceptions_differ
                        and not schemas_differ
                        and not text_differs
                    ):
                        combined = [SemanticEditClass.IDENTICAL.value]
                    elif (
                        text_differs
                        and not files_differ
                        and not symbols_differ
                        and not interfaces_differ
                        and not side_effects_differ
                        and not exceptions_differ
                        and not schemas_differ
                    ):
                        combined = [SemanticEditClass.EQUIVALENT_REFORMAT.value]
                    else:
                        combined = [SemanticEditClass.UNKNOWN.value]
                ast_edit_classes = tuple(combined)
            if ast_edit_classes:
                bases.add(OutcomeClassificationBasis.AST_EDIT_CLASSES.value)
    else:
        # No structural evidence: text_differs from patch CIDs only when both set.
        if text_differs:
            bases.add(OutcomeClassificationBasis.TEXT_DIFF.value)
        if compressed.patch_cid is None and expanded.patch_cid is None:
            ast_edit_classes = (SemanticEditClass.UNKNOWN.value,)
        elif (
            compressed.patch_cid is not None
            and expanded.patch_cid is not None
            and compressed.patch_cid == expanded.patch_cid
        ):
            ast_edit_classes = (SemanticEditClass.IDENTICAL.value,)
            bases.add(OutcomeClassificationBasis.AST_EDIT_CLASSES.value)
        else:
            # Distinct patches without structural evidence → unknown, not failure.
            ast_edit_classes = (SemanticEditClass.UNKNOWN.value,)
            bases.add(OutcomeClassificationBasis.AST_EDIT_CLASSES.value)

    if force_human_review or (
        compressed.acceptance_disposition
        == AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value
        or expanded.acceptance_disposition
        == AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value
    ):
        bases.add(OutcomeClassificationBasis.HUMAN_REVIEW.value)
        human_review_required = True
    else:
        human_review_required = False

    # Always record verification receipts when both sides present.
    bases.add(OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value)

    semantic_equivalent = _classify_semantic_equivalence(
        text_differs=text_differs,
        files_differ=files_differ,
        symbols_differ=symbols_differ,
        interfaces_differ=interfaces_differ,
        side_effects_differ=side_effects_differ,
        exceptions_differ=exceptions_differ,
        schemas_differ=schemas_differ,
        tests_differ=tests_differ,
        proofs_differ=proofs_differ,
        counterexamples_differ=counterexamples_differ,
        static_analysis_differ=static_analysis_differ,
        acceptance_differ=acceptance_differ,
        ast_edit_classes=ast_edit_classes,
        structural_present=structural_present,
        compressed_bucket=_attempt_bucket(compressed),
        expanded_bucket=_attempt_bucket(expanded),
        classification_bases=frozenset(bases),
        compressed_patch_cid=compressed.patch_cid,
        expanded_patch_cid=expanded.patch_cid,
    )

    return _DiffFlags(
        text_differs=text_differs,
        files_differ=files_differ,
        symbols_differ=symbols_differ,
        interfaces_differ=interfaces_differ,
        side_effects_differ=side_effects_differ,
        exceptions_differ=exceptions_differ,
        schemas_differ=schemas_differ,
        tests_differ=tests_differ,
        proofs_differ=proofs_differ,
        counterexamples_differ=counterexamples_differ,
        static_analysis_differ=static_analysis_differ,
        performance_differ=performance_differ,
        acceptance_differ=acceptance_differ,
        human_review_required=human_review_required,
        ast_edit_classes=ast_edit_classes,
        semantic_equivalent=semantic_equivalent,
        classification_bases=tuple(sorted(bases)),
        structural_evidence_present=structural_present,
    )


def _classify_semantic_equivalence(
    *,
    text_differs: bool,
    files_differ: bool,
    symbols_differ: bool,
    interfaces_differ: bool,
    side_effects_differ: bool,
    exceptions_differ: bool,
    schemas_differ: bool,
    tests_differ: bool,
    proofs_differ: bool,
    counterexamples_differ: bool,
    static_analysis_differ: bool,
    acceptance_differ: bool,
    ast_edit_classes: Sequence[str],
    structural_present: bool,
    compressed_bucket: str,
    expanded_bucket: str,
    classification_bases: frozenset[str],
    compressed_patch_cid: str | None,
    expanded_patch_cid: str | None,
) -> bool | None:
    """Decide semantic_equivalent using non-text evidence.

    Returns:
      * ``True``  — non-text evidence supports equivalence of valid outcomes
      * ``False`` — non-text evidence supports divergence
      * ``None``  — insufficient non-text evidence (text alone never decides)
    """

    # Different success/failure buckets cannot be semantically equivalent.
    if compressed_bucket != expanded_bucket:
        if compressed_bucket == "inconclusive" or expanded_bucket == "inconclusive":
            return None
        # Both decisive but different → not equivalent (verification evidence).
        if OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value in classification_bases:
            return False
        return None

    # Verification divergences on tests/proofs/counterexamples are non-text.
    if (
        tests_differ
        or proofs_differ
        or counterexamples_differ
        or static_analysis_differ
        or acceptance_differ
    ):
        return False

    structural_diverge = (
        files_differ
        or symbols_differ
        or interfaces_differ
        or side_effects_differ
        or exceptions_differ
        or schemas_differ
    )
    edit_set = set(ast_edit_classes)
    divergent_edits = edit_set & _DIVERGENT_EDITS
    unknown_edits = SemanticEditClass.UNKNOWN.value in edit_set

    if divergent_edits or structural_diverge:
        return False

    if compressed_bucket == "success" and expanded_bucket == "success":
        # Both succeeded with no verification/structural divergence.
        equivalence_preserving = bool(edit_set) and edit_set <= (
            _EQUIVALENCE_PRESERVING_EDITS
        )
        identical_patches = (
            compressed_patch_cid is not None
            and expanded_patch_cid is not None
            and compressed_patch_cid == expanded_patch_cid
        )
        if identical_patches:
            return True
        if structural_present and equivalence_preserving:
            # Text may still differ (reformat/rename/reorder).
            return True
        if structural_present and not unknown_edits and not structural_diverge:
            # Structural sets match and no divergent edits.
            return True
        if (
            not structural_present
            and not text_differs
            and identical_patches is False
            and compressed_patch_cid is None
            and expanded_patch_cid is None
        ):
            # No patches and no structural evidence — cannot claim equivalence.
            return None
        if not structural_present and not text_differs and identical_patches:
            return True
        if not structural_present and text_differs:
            # Text differs without structural evidence → inconclusive equivalence.
            return None
        if unknown_edits and not structural_present:
            return None
        if equivalence_preserving:
            return True
        # Default when both success and no diverge signals: equivalent when
        # verification bases are present and no structural diverge.
        if OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value in classification_bases:
            if not structural_diverge and not divergent_edits:
                # Without structural proof of sameness, only claim equivalence
                # when patches match or only equivalence-preserving edits.
                if identical_patches or equivalence_preserving:
                    return True
                # Matching verification with no structural evidence of diverge
                # and identical empty structural projection:
                if structural_present:
                    return True
                return None
        return None

    if compressed_bucket == "failure" and expanded_bucket == "failure":
        # Same failure bucket without structural diverge is not "equivalent success".
        return False

    return None


# ---------------------------------------------------------------------------
# Comparative outcome classification
# ---------------------------------------------------------------------------


def _classify_comparative_outcome(
    *,
    compressed: PairedAttemptRecord,
    expanded: PairedAttemptRecord | None,
    compressed_verification: VerificationProjection,
    expanded_verification: VerificationProjection | None,
    flags: _DiffFlags,
) -> str:
    """Return a closed ComparativeOutcome value (fail-closed priority order)."""

    # Human review takes precedence when forced or disposition requires it.
    if flags.human_review_required:
        return ComparativeOutcome.HUMAN_REVIEW_REQUIRED.value

    if expanded is None or expanded_verification is None:
        return ComparativeOutcome.HUMAN_REVIEW_REQUIRED.value

    c_bucket = _attempt_bucket(compressed)
    e_bucket = _attempt_bucket(expanded)

    # Inconclusive verification / attempt status must stay inconclusive.
    if c_bucket == "inconclusive" or e_bucket == "inconclusive":
        return ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value
    if _verification_is_inconclusive(compressed_verification) or (
        _verification_is_inconclusive(expanded_verification)
        and e_bucket != "failure"
    ):
        # Expanded with explicit failure + counterexample is decisive even if
        # some matrix flags are null; only gate pure incomplete evidence.
        if e_bucket == "success" or c_bucket == "success":
            if _verification_is_inconclusive(
                compressed_verification
            ) or _verification_is_inconclusive(expanded_verification):
                return ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value

    # Distinct asymmetric failure/success outcomes.
    if c_bucket == "failure" and e_bucket == "success":
        return ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value
    if c_bucket == "success" and e_bucket == "failure":
        return ComparativeOutcome.COMPRESSED_SUCCEEDED_EXPANDED_FAILED.value

    if c_bucket == "failure" and e_bucket == "failure":
        c_reasons = frozenset(compressed.failure_reason_codes)
        e_reasons = frozenset(expanded.failure_reason_codes)
        if c_reasons and e_reasons and c_reasons == e_reasons:
            return ComparativeOutcome.BOTH_FAILED_SAME_REASON.value
        return ComparativeOutcome.BOTH_FAILED_DIFFERENT_REASON.value

    # Both success path.
    if c_bucket == "success" and e_bucket == "success":
        if flags.semantic_equivalent is True:
            return ComparativeOutcome.EQUIVALENT_SUCCESS.value

        c_score = _score_verification(compressed_verification)
        e_score = _score_verification(expanded_verification)
        # Relative quality only when verification differs with non-text bases.
        if (
            flags.tests_differ
            or flags.proofs_differ
            or flags.counterexamples_differ
            or flags.static_analysis_differ
            or flags.acceptance_differ
        ):
            if e_score > c_score:
                return ComparativeOutcome.EXPANDED_BETTER.value
            if c_score > e_score:
                return ComparativeOutcome.COMPRESSED_BETTER.value

        if flags.semantic_equivalent is False:
            return ComparativeOutcome.BOTH_VALID_DIFFERENT.value

        # Equivalence unknown: both valid, not classified as equivalent.
        return ComparativeOutcome.BOTH_VALID_DIFFERENT.value

    # Fallback — should be unreachable given bucket partition.
    return ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value


def _failure_classified_for(outcome: str) -> bool:
    return outcome in _FAILURE_LIKE_OUTCOMES


def _ensure_non_text_bases_for_failure(
    outcome: str, bases: Sequence[str]
) -> tuple[str, ...]:
    """Guarantee failure-like outcomes carry non-text classification bases."""

    failure_like = _failure_classified_for(outcome)
    base_set = set(bases)
    if failure_like and not non_text_classification_bases(base_set):
        # Inject verification receipts — attempt status comparison is verification
        # evidence, never pure text.
        base_set.add(OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value)
    if failure_like:
        assert_failure_classification_not_text_alone(
            base_set, failure_classified=True, name="classification_bases"
        )
    return tuple(sorted(base_set))


# ---------------------------------------------------------------------------
# Header construction
# ---------------------------------------------------------------------------


def _build_report_header(
    *,
    seed: GovernorArtifactHeader | None,
    plan_cid: str,
    shadow_result_cid: str,
    compressed: PairedAttemptRecord,
    execution_mode: str,
    artifact_kind: str,
) -> GovernorArtifactHeader:
    mode = execution_mode
    try:
        mode = ExecutionMode(execution_mode).value
    except (TypeError, ValueError) as exc:
        raise SemanticGovernorDifferentialError(
            f"execution_mode has unsupported value {execution_mode!r}"
        ) from exc

    if seed is not None:
        repository_state_cid = seed.repository_state_cid
        context_pack_cid = seed.context_pack_cid
        verification_bundle_cid = (
            seed.verification_bundle_cid
            or compressed.verification.verification_bundle_cid
        )
        policy_cid = seed.provenance.policy_cid
        producer_version = seed.provenance.producer_version
    else:
        repository_state_cid = compressed.context_pack_cid
        context_pack_cid = compressed.context_pack_cid
        verification_bundle_cid = compressed.verification.verification_bundle_cid
        policy_cid = compressed.verification.verification_bundle_cid
        producer_version = "1"

    terminal = (
        GovernorTerminalStatus.SIMULATED.value
        if mode == ExecutionMode.SIMULATED.value
        else GovernorTerminalStatus.COMPLETE.value
    )

    generator = GeneratorIdentity(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        interface_id=COMPARE_SHADOW_RESULTS_INTERFACE,
    )
    provenance = ArtifactProvenance(
        producer_id="semantic_governor",
        producer_version=producer_version,
        execution_mode=mode,
        authority_source=AuthoritySource.DETERMINISTIC,
        input_cids=(plan_cid, shadow_result_cid, compressed.context_pack_cid),
        tool_ids=("differential.v1", "compare_shadow_results.v1"),
        policy_cid=policy_cid,
        notes=None,
    )
    return GovernorArtifactHeader(
        artifact_kind=artifact_kind,
        repository_state_cid=repository_state_cid,
        context_pack_cid=context_pack_cid,
        verification_bundle_cid=verification_bundle_cid,
        generator=generator,
        provenance=provenance,
        terminal_status=terminal,
        assumptions=(
            GovernorAssumption(
                assumption_id="text_not_semantic_failure",
                kind=AssumptionKind.VERIFICATION,
                statement=(
                    "Textual difference alone cannot classify semantic failure; "
                    "structural and verification evidence are required"
                ),
                supporting_cids=(plan_cid,),
            ),
            GovernorAssumption(
                assumption_id="expanded_oracle_only",
                kind=AssumptionKind.VERIFICATION,
                statement=(
                    "Expanded shadow output remains oracle/candidate only"
                ),
                supporting_cids=(plan_cid,),
            ),
        ),
        metadata={
            "evidence": SCG_DIFFERENTIAL_EVIDENCE,
            "plan_cid": plan_cid,
            "shadow_result_cid": shadow_result_cid,
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_shadow_results(
    compressed_result: (
        PairedAttemptRecord | Mapping[str, Any] | ShadowExecutionResult | None
    ) = None,
    expanded_result: PairedAttemptRecord | Mapping[str, Any] | None = None,
    verification_evidence: Any | None = None,
    *,
    shadow_result: ShadowExecutionResult | Mapping[str, Any] | None = None,
    structural_evidence: StructuralComparisonEvidence | Mapping[str, Any] | None = None,
    compressed_verification_evidence: Any | None = None,
    expanded_verification_evidence: Any | None = None,
    plan_cid: str | None = None,
    shadow_result_cid: str | None = None,
    header_seed: GovernorArtifactHeader | Mapping[str, Any] | None = None,
    human_review_required: bool | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SemanticDifferentialOutcome:
    """Compare paired shadow patch outcomes beyond textual equality.

    Parameters
    ----------
    compressed_result:
        Compressed attempt record, mapping, or a full ``ShadowExecutionResult``
        (when ``expanded_result`` is omitted the expanded attempt is taken from
        the shadow result).
    expanded_result:
        Expanded attempt record or mapping (oracle / candidate only).
    verification_evidence:
        Optional shared verification evidence applied to both sides when
        side-specific evidence is not provided.  Never upgrades status.
    shadow_result:
        Optional full paired shadow execution result.  When supplied, plan and
        result CIDs are taken from it and attempts default from it.
    structural_evidence:
        Optional file/symbol/AST/interface/effect/exception/schema projections.
    """

    sealed_shadow: ShadowExecutionResult | None = None
    if shadow_result is not None:
        sealed_shadow = _require_shadow_result(shadow_result)
    elif isinstance(compressed_result, ShadowExecutionResult):
        sealed_shadow = _require_shadow_result(compressed_result)
        compressed_result = None
    elif isinstance(compressed_result, Mapping) and (
        compressed_result.get("interface_id")
        == "ShadowExecutionResult@1"
        or compressed_result.get("schema", "").endswith("shadow-execution-result@1")
    ):
        sealed_shadow = _require_shadow_result(compressed_result)
        compressed_result = None

    if sealed_shadow is not None:
        compressed = sealed_shadow.compressed_attempt
        expanded = sealed_shadow.expanded_attempt
        resolved_plan_cid = sealed_shadow.plan_cid
        resolved_shadow_cid = sealed_shadow.result_cid
        seed_header = sealed_shadow.header
        execution_mode = sealed_shadow.header.provenance.execution_mode
    else:
        if compressed_result is None:
            raise SemanticGovernorDifferentialError(
                "compressed_result or shadow_result is required",
                reason_code="missing_compressed",
            )
        compressed = _require_attempt(
            compressed_result,
            expected_role=ShadowAttemptRole.COMPRESSED.value,
            name="compressed_result",
        )
        if expanded_result is None:
            expanded = None
        else:
            expanded = _require_attempt(
                expanded_result,
                expected_role=ShadowAttemptRole.EXPANDED.value,
                name="expanded_result",
            )
        if plan_cid is None:
            raise SemanticGovernorDifferentialError(
                "plan_cid is required when shadow_result is not provided",
                reason_code="missing_plan_cid",
            )
        if shadow_result_cid is None:
            raise SemanticGovernorDifferentialError(
                "shadow_result_cid is required when shadow_result is not provided",
                reason_code="missing_shadow_result_cid",
            )
        resolved_plan_cid = _cid(plan_cid, "plan_cid")
        resolved_shadow_cid = _cid(shadow_result_cid, "shadow_result_cid")
        seed_header = None
        execution_mode = compressed.execution_mode

    # Explicit attempt overrides when shadow_result was also provided.
    if sealed_shadow is not None and compressed_result is not None:
        compressed = _require_attempt(
            compressed_result,
            expected_role=ShadowAttemptRole.COMPRESSED.value,
            name="compressed_result",
        )
    if sealed_shadow is not None and expanded_result is not None:
        expanded = _require_attempt(
            expanded_result,
            expected_role=ShadowAttemptRole.EXPANDED.value,
            name="expanded_result",
        )

    if plan_cid is not None and sealed_shadow is not None:
        # Allow explicit plan_cid only when it matches the sealed result.
        explicit = _cid(plan_cid, "plan_cid")
        if explicit != resolved_plan_cid:
            raise SemanticGovernorDifferentialError(
                "plan_cid does not match shadow_result.plan_cid",
                reason_code="plan_cid_mismatch",
            )
    if shadow_result_cid is not None and sealed_shadow is not None:
        explicit = _cid(shadow_result_cid, "shadow_result_cid")
        if explicit != resolved_shadow_cid:
            raise SemanticGovernorDifferentialError(
                "shadow_result_cid does not match shadow_result.result_cid",
                reason_code="shadow_result_cid_mismatch",
            )

    if expanded is not None:
        assert_expanded_never_accepted(
            expanded.acceptance_disposition, role=expanded.role
        )
        if expanded.verification.production_eligible:
            raise SemanticGovernorDifferentialError(
                "expanded attempt cannot be production_eligible",
                reason_code="expanded_production_eligible",
            )

    # Resolve verification projections (never upgrade).
    c_ver = _verification_from_evidence(
        compressed_verification_evidence
        if compressed_verification_evidence is not None
        else verification_evidence,
        fallback=compressed.verification,
    )
    e_ver: VerificationProjection | None
    if expanded is None:
        e_ver = None
    else:
        e_ver = _verification_from_evidence(
            expanded_verification_evidence
            if expanded_verification_evidence is not None
            else verification_evidence,
            fallback=expanded.verification,
        )

    structural = _require_structural(structural_evidence)

    force_review = bool(human_review_required) if human_review_required is not None else False
    if header_seed is not None:
        if isinstance(header_seed, GovernorArtifactHeader):
            seed_header = header_seed
        elif isinstance(header_seed, Mapping):
            seed_header = GovernorArtifactHeader.from_dict(header_seed)
        else:
            raise SemanticGovernorDifferentialError(
                "header_seed must be GovernorArtifactHeader or mapping"
            )

    flags = _project_diff_flags(
        compressed=compressed,
        expanded=expanded,
        compressed_verification=c_ver,
        expanded_verification=e_ver,
        structural=structural,
        force_human_review=force_review,
    )

    outcome = _classify_comparative_outcome(
        compressed=compressed,
        expanded=expanded,
        compressed_verification=c_ver,
        expanded_verification=e_ver,
        flags=flags,
    )

    classification_bases = _ensure_non_text_bases_for_failure(
        outcome, flags.classification_bases
    )
    failure_classified = _failure_classified_for(outcome)

    # Equivalence cannot be True when failure is classified.
    semantic_equivalent = flags.semantic_equivalent
    if failure_classified and semantic_equivalent is True:
        semantic_equivalent = False

    # Build headers / sealed artifacts.
    report_header = _build_report_header(
        seed=seed_header,
        plan_cid=resolved_plan_cid,
        shadow_result_cid=resolved_shadow_cid,
        compressed=compressed,
        execution_mode=execution_mode,
        artifact_kind="differential_patch_report",
    )
    comparison_header = _build_report_header(
        seed=seed_header,
        plan_cid=resolved_plan_cid,
        shadow_result_cid=resolved_shadow_cid,
        compressed=compressed,
        execution_mode=execution_mode,
        artifact_kind="semantic_outcome_comparison",
    )

    c_cost = compressed.cost_timing
    if expanded is not None:
        e_cost = expanded.cost_timing
        e_in = e_cost.input_tokens
        e_out = e_cost.output_tokens
        e_wall = e_cost.wall_time_ms
        e_spend = e_cost.model_spend_micros
        expanded_acceptance = expanded.acceptance_disposition
    else:
        e_in = 0
        e_out = 0
        e_wall = 0
        e_spend = 0
        expanded_acceptance = AcceptanceDisposition.NOT_ACCEPTED.value

    report = DifferentialPatchReport(
        header=report_header,
        plan_cid=resolved_plan_cid,
        shadow_result_cid=resolved_shadow_cid,
        text_differs=flags.text_differs,
        files_differ=flags.files_differ,
        symbols_differ=flags.symbols_differ,
        interfaces_differ=flags.interfaces_differ,
        side_effects_differ=flags.side_effects_differ,
        exceptions_differ=flags.exceptions_differ,
        schemas_differ=flags.schemas_differ,
        tests_differ=flags.tests_differ,
        proofs_differ=flags.proofs_differ,
        counterexamples_differ=flags.counterexamples_differ,
        static_analysis_differ=flags.static_analysis_differ,
        performance_differ=flags.performance_differ,
        acceptance_differ=flags.acceptance_differ,
        human_review_required=flags.human_review_required
        or outcome == ComparativeOutcome.HUMAN_REVIEW_REQUIRED.value,
        ast_edit_classes=flags.ast_edit_classes,
        compressed_input_tokens=c_cost.input_tokens,
        expanded_input_tokens=e_in,
        compressed_output_tokens=c_cost.output_tokens,
        expanded_output_tokens=e_out,
        compressed_wall_time_ms=c_cost.wall_time_ms,
        expanded_wall_time_ms=e_wall,
        compressed_model_spend_micros=c_cost.model_spend_micros,
        expanded_model_spend_micros=e_spend,
        semantic_equivalent=semantic_equivalent,
        failure_classified=failure_classified,
        classification_bases=classification_bases,
        textual_difference_is_not_semantic_failure=True,
        metadata={
            "evidence": SCG_DIFFERENTIAL_EVIDENCE,
            "structural_evidence_present": flags.structural_evidence_present,
            "comparative_outcome": outcome,
        },
    )
    verify_report_identity(report)

    # Compressed acceptance: never promote beyond the attempt disposition.
    compressed_acceptance = compressed.acceptance_disposition
    if outcome == ComparativeOutcome.HUMAN_REVIEW_REQUIRED.value:
        if compressed_acceptance not in {
            AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value,
            AcceptanceDisposition.NOT_ACCEPTED.value,
        }:
            compressed_acceptance = AcceptanceDisposition.NOT_ACCEPTED.value

    if expanded is not None:
        assert_expanded_never_accepted(expanded_acceptance, role=expanded.role)

    comparison = SemanticOutcomeComparison(
        header=comparison_header,
        plan_cid=resolved_plan_cid,
        shadow_result_cid=resolved_shadow_cid,
        differential_report_cid=report.report_cid,
        comparative_outcome=outcome,
        compressed_acceptance=compressed_acceptance,
        expanded_acceptance=expanded_acceptance,
        human_review_required=report.human_review_required,
        classification_bases=classification_bases,
        notes=_optional_text(notes, "notes"),
        metadata={
            "evidence": SCG_DIFFERENTIAL_EVIDENCE,
            "semantic_equivalent": semantic_equivalent,
            "failure_classified": failure_classified,
        },
    )
    verify_comparison_identity(comparison)

    outcome_meta: dict[str, Any] = {
        "evidence": SCG_DIFFERENTIAL_EVIDENCE,
        "interface_id": COMPARE_SHADOW_RESULTS_INTERFACE,
    }
    if metadata:
        thawed = _mapping(metadata, "metadata")
        outcome_meta.update(_thaw_structured(thawed))

    return SemanticDifferentialOutcome(
        report=report,
        comparison=comparison,
        metadata=outcome_meta,
    )


def classify_comparative_outcome(
    compressed_result: PairedAttemptRecord | Mapping[str, Any],
    expanded_result: PairedAttemptRecord | Mapping[str, Any] | None,
    *,
    structural_evidence: StructuralComparisonEvidence | Mapping[str, Any] | None = None,
    human_review_required: bool = False,
) -> str:
    """Classify a comparative outcome without sealing durable artifacts.

    Useful for lightweight routing decisions.  Full sealed comparison should
    use :func:`compare_shadow_results`.
    """

    compressed = _require_attempt(
        compressed_result,
        expected_role=ShadowAttemptRole.COMPRESSED.value,
        name="compressed_result",
    )
    expanded: PairedAttemptRecord | None
    if expanded_result is None:
        expanded = None
        e_ver = None
    else:
        expanded = _require_attempt(
            expanded_result,
            expected_role=ShadowAttemptRole.EXPANDED.value,
            name="expanded_result",
        )
        e_ver = expanded.verification
    structural = _require_structural(structural_evidence)
    flags = _project_diff_flags(
        compressed=compressed,
        expanded=expanded,
        compressed_verification=compressed.verification,
        expanded_verification=e_ver,
        structural=structural,
        force_human_review=human_review_required,
    )
    return _classify_comparative_outcome(
        compressed=compressed,
        expanded=expanded,
        compressed_verification=compressed.verification,
        expanded_verification=e_ver,
        flags=flags,
    )


__all__ = [
    "COMPARE_SHADOW_RESULTS_INTERFACE",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "SCG_DIFFERENTIAL_EVIDENCE",
    "SEMANTIC_DIFFERENTIAL_OUTCOME_INTERFACE",
    "SEMANTIC_DIFFERENTIAL_OUTCOME_SCHEMA",
    "STRUCTURAL_COMPARISON_EVIDENCE_SCHEMA",
    "STRUCTURAL_PROJECTION_SCHEMA",
    "AttemptStructuralProjection",
    "SemanticDifferentialOutcome",
    "SemanticGovernorDifferentialError",
    "StructuralComparisonEvidence",
    "classify_comparative_outcome",
    "compare_shadow_results",
]
