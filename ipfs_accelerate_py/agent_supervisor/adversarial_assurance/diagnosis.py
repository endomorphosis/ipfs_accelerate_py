"""Orchestrate bounded counterexample minimization and survivor diagnosis (AAE-045).

Interface surface:

* ``diagnose_surviving_mutant@1`` — minimize survivor diagnostics with the
  released ``CounterexampleMinimizer@1``, seal a bounded
  ``SurvivingMutantReport@1``, run semantic nine-step diagnosis, and always
  persist an ``AssuranceGap@1`` for every high-risk survivor.

Normative properties (acceptance):

* Existing ``CounterexampleMinimizer`` and semantic diagnostics produce
  bounded reproductions (no full-log bodies in model/report context).
* Minimization failure is explicit (``minimization_failed`` + failure reason +
  bounded log digest), never silent success.
* Every high-risk survivor always persists an ``AssuranceGap``.
* Human review accompanies an unknown gap rather than replacing it.
* No production policy change; cold import is side-effect free.

This module composes released authorities:

* AAE-031 minimized survivor reports / bounded reproduction evidence
* AAE-030 nine-step ``diagnose_surviving_mutant`` (datasets pure path)
* AAE-028 ``classify_assurance_gap``
* IVP-011 ``CounterexampleMinimizer`` / ``minimize_counterexample``
* AAE-044 survivor outcomes as diagnosis inputs
* Optional ``AssuranceGapRepository@1`` for durable gap persistence
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

from ipfs_accelerate_py.agent_supervisor.verification.counterexamples import (
    COUNTEREXAMPLE_MINIMIZER_INTERFACE,
    CounterexampleMinimizationError,
    CounterexampleMinimizationResult,
    CounterexampleMinimizer,
    FailureMaterial,
    MinimizationBudget,
    MinimizationGuarantee,
    MinimizationQuality,
    MinimizationRequest,
    RerunOracle,
    minimize_counterexample,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.analysis_contracts import (
    AnalysisContractError,
    AssuranceGap,
    AssuranceGapClass,
    GapSeverity,
    MinimizedEvidenceBinding,
    SourceSpan,
    SurvivingMutantReport,
    SurvivorRiskClass,
    verify_gap_identity,
    verify_survivor_report_identity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    AssuranceArtifactHeader,
    AssuranceBaseError,
    GeneratorIdentity,
    VersionBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.diagnosis import (
    DIAGNOSE_SURVIVING_MUTANT_INTERFACE as DATASETS_DIAGNOSE_INTERFACE,
    DIAGNOSIS_STEP_ORDER,
    DiagnosisDisposition,
    DiagnosisError as SemanticDiagnosisError,
    DiagnosisMutationBinding,
    DiagnosisOutcomeBinding,
    DiagnosisSignals,
    DiagnosisStepId,
    SurvivorDiagnosis,
    diagnose_surviving_mutant as semantic_diagnose_surviving_mutant,
    verify_survivor_diagnosis_identity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    EquivalenceAssessmentStatus,
    MutationOutcomeStatus,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.gaps import (
    AssuranceGapCause,
    CLASSIFY_ASSURANCE_GAP_INTERFACE,
    DetectionComparisonResult,
    GapClassificationError,
    GapClassificationSubject,
    classify_assurance_gap,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.minimization import (
    BUILD_SURVIVING_MUTANT_REPORT_INTERFACE,
    BoundedLogDigest,
    MinimizationError as SurvivorMinimizationError,
    MinimizationStatus,
    SurvivorMinimizationSubject,
    SurvivorReportBuildResult,
    build_surviving_mutant_report,
    logs_remain_bounded,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_bytes,
    cid_for_structured,
    validate_cid,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

DIAGNOSE_SURVIVING_MUTANT_INTERFACE: Final[str] = "diagnose_surviving_mutant@1"

SURVIVOR_DIAGNOSIS_RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "survivor-diagnosis-run@1"
)
SURVIVOR_DIAGNOSIS_RUN_INTERFACE: Final[str] = "SurvivorDiagnosisRun@1"
BOUNDED_REPRODUCTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "bounded-reproduction@1"
)
GAP_PERSIST_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "gap-persist-request@1"
)

AAE_DIAGNOSIS_RUN_EVIDENCE: Final[str] = "aae/diagnosis-run@1"
ADAPTER_ID: Final[str] = "aae-diagnosis-orchestration"
BOARD_NAMESPACE: Final[str] = "adversarial-assurance-engine-v1"
GENERATOR_ID: Final[str] = "diagnosis_orchestration"
GENERATOR_VERSION: Final[str] = "1.0.0"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_LIST: Final[int] = 1_024
MAX_DIAGNOSTIC: Final[int] = 1_024
MAX_ARGV: Final[int] = 256
DEFAULT_LOG_DIGEST_BYTES: Final[int] = 4_096

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

# High-risk survivors always require a durable AssuranceGap (plan §15 / AAE-045).
DEFAULT_HIGH_RISK_CLASSES: Final[tuple[str, ...]] = (
    SurvivorRiskClass.CRITICAL_SECURITY.value,
    SurvivorRiskClass.AUTHORIZATION.value,
    SurvivorRiskClass.FINANCIAL_LEGAL.value,
    SurvivorRiskClass.DURABILITY.value,
    SurvivorRiskClass.DISTRIBUTED_TRANSITION.value,
    SurvivorRiskClass.PROOF_RECEIPT_TRUST.value,
    SurvivorRiskClass.CRITICAL_INVARIANT.value,
    SurvivorRiskClass.HIGH.value,
)

_RISK_TO_SEVERITY: Final[Mapping[str, str]] = MappingProxyType(
    {
        SurvivorRiskClass.CRITICAL_SECURITY.value: GapSeverity.CRITICAL.value,
        SurvivorRiskClass.AUTHORIZATION.value: GapSeverity.CRITICAL.value,
        SurvivorRiskClass.FINANCIAL_LEGAL.value: GapSeverity.CRITICAL.value,
        SurvivorRiskClass.DURABILITY.value: GapSeverity.HIGH.value,
        SurvivorRiskClass.DISTRIBUTED_TRANSITION.value: GapSeverity.HIGH.value,
        SurvivorRiskClass.PROOF_RECEIPT_TRUST.value: GapSeverity.HIGH.value,
        SurvivorRiskClass.CRITICAL_INVARIANT.value: GapSeverity.HIGH.value,
        SurvivorRiskClass.HIGH.value: GapSeverity.HIGH.value,
        SurvivorRiskClass.MEDIUM.value: GapSeverity.MEDIUM.value,
        SurvivorRiskClass.LOCAL_BUG.value: GapSeverity.MEDIUM.value,
        SurvivorRiskClass.LOW.value: GapSeverity.LOW.value,
    }
)

REASON_BOUNDED_REPRODUCTION: Final[str] = "bounded_reproduction"
REASON_MINIMIZATION_SUCCEEDED: Final[str] = "minimization_succeeded"
REASON_MINIMIZATION_FAILED: Final[str] = "minimization_failed_explicit"
REASON_MINIMIZATION_SKIPPED: Final[str] = "minimization_skipped_prebuilt"
REASON_SEMANTIC_DIAGNOSIS: Final[str] = "semantic_diagnosis_complete"
REASON_HIGH_RISK_GAP_REQUIRED: Final[str] = "high_risk_gap_required"
REASON_GAP_PERSISTED: Final[str] = "assurance_gap_persisted"
REASON_GAP_IN_MEMORY: Final[str] = "assurance_gap_sealed"
REASON_UNKNOWN_GAP_HUMAN_REVIEW: Final[str] = (
    "unknown_gap_requires_human_review"
)
REASON_HUMAN_REVIEW_ACCOMPANIES_GAP: Final[str] = (
    "human_review_accompanies_gap"
)
REASON_NO_PRODUCTION_POLICY_CHANGE: Final[str] = "production_policy_unchanged"
REASON_LOGS_BOUNDED: Final[str] = "logs_remain_bounded"
REASON_COUNTEREXAMPLE_MINIMIZER: Final[str] = "counterexample_minimizer_used"


# ---------------------------------------------------------------------------
# Errors and closed enums
# ---------------------------------------------------------------------------


class DiagnosisRuntimeError(ValueError):
    """Raised when diagnosis orchestration fails closed."""

    def __init__(self, message: str, *, reason_code: str = "malformed_input") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class HighRiskGapMissingError(DiagnosisRuntimeError):
    """Raised when a high-risk survivor would omit an AssuranceGap."""


class DiagnosisPhase(str, Enum):
    """Ordered phases recorded on a diagnosis run."""

    ADMIT = "admit"
    MINIMIZE = "minimize"
    SURVIVOR_REPORT = "survivor_report"
    SEMANTIC_DIAGNOSIS = "semantic_diagnosis"
    GAP = "gap"
    PERSIST = "persist"


class GapPersistStatus(str, Enum):
    """Closed gap-persistence disposition."""

    NOT_REQUIRED = "not_required"
    SEALED_ONLY = "sealed_only"
    PERSISTED = "persisted"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Optional durable gap repository (AAE-036 surface, fail-closed protocol)
# ---------------------------------------------------------------------------


class GapPersistResultLike(Protocol):
    """Minimal durable gap persist result surface."""

    gap_cid: str


class AssuranceGapRepository(Protocol):
    """Closed durable assurance-gap repository surface (``AssuranceGapRepository@1``)."""

    def persist_gap(
        self,
        workspace: str,
        payload: Mapping[str, Any],
        *,
        expected_cid: str,
        artifact_operation_id: str,
        history_operation_id: str,
        expected_history_generation: int,
        expected_history_head_cid: str | None,
        replicate: bool = False,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class GapPersistRequest:
    """Caller-supplied durable-persist coordinates for one AssuranceGap."""

    schema: str = GAP_PERSIST_REQUEST_SCHEMA
    workspace: str = "default"
    artifact_operation_id: str = "gap_artifact_op"
    history_operation_id: str = "gap_history_op"
    expected_history_generation: int = 0
    expected_history_head_cid: str | None = None
    replicate: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "workspace", _text(self.workspace, "workspace")
        )
        object.__setattr__(
            self,
            "artifact_operation_id",
            _token(self.artifact_operation_id, "artifact_operation_id"),
        )
        object.__setattr__(
            self,
            "history_operation_id",
            _token(self.history_operation_id, "history_operation_id"),
        )
        gen = self.expected_history_generation
        if type(gen) is not int or isinstance(gen, bool) or gen < 0:
            raise DiagnosisRuntimeError(
                "expected_history_generation must be a nonnegative integer"
            )
        if self.expected_history_head_cid is not None:
            object.__setattr__(
                self,
                "expected_history_head_cid",
                _cid(
                    self.expected_history_head_cid,
                    "expected_history_head_cid",
                ),
            )
        object.__setattr__(
            self, "replicate", _bool(self.replicate, "replicate")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "workspace": self.workspace,
            "artifact_operation_id": self.artifact_operation_id,
            "history_operation_id": self.history_operation_id,
            "expected_history_generation": self.expected_history_generation,
            "expected_history_head_cid": self.expected_history_head_cid,
            "replicate": self.replicate,
        }


@dataclass
class InMemoryAssuranceGapRepository:
    """Test/dev gap repository that seals gaps without durable coordination."""

    gaps: MutableMapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def persist_gap(
        self,
        workspace: str,
        payload: Mapping[str, Any],
        *,
        expected_cid: str,
        artifact_operation_id: str,
        history_operation_id: str,
        expected_history_generation: int,
        expected_history_head_cid: str | None,
        replicate: bool = False,
    ) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise DiagnosisRuntimeError("gap payload must be a mapping")
        sealed = AssuranceGap.from_dict(payload)
        actual = sealed.gap_cid
        if actual != expected_cid:
            raise DiagnosisRuntimeError(
                f"gap cid mismatch: expected {expected_cid!r} got {actual!r}",
                reason_code="gap_cid_mismatch",
            )
        key = f"{workspace}:{actual}"
        record = {
            "gap_cid": actual,
            "workspace": workspace,
            "artifact_operation_id": artifact_operation_id,
            "history_operation_id": history_operation_id,
            "expected_history_generation": expected_history_generation,
            "expected_history_head_cid": expected_history_head_cid,
            "replicate": bool(replicate),
            "payload": sealed.to_dict(),
        }
        self.gaps[key] = MappingProxyType(dict(record))
        return MappingProxyType({"gap_cid": actual, "workspace": workspace})


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise DiagnosisRuntimeError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise DiagnosisRuntimeError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise DiagnosisRuntimeError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, empty=False)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise DiagnosisRuntimeError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise DiagnosisRuntimeError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise DiagnosisRuntimeError(f"{name} must be a boolean")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise DiagnosisRuntimeError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _stable_unique(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        if isinstance(value, enum_type):
            return value.value  # type: ignore[return-value]
        return enum_type(value).value  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise DiagnosisRuntimeError(
            f"{name} has unsupported value {value!r}"
        ) from exc


def _risk_value(value: Any) -> str:
    return _enum(value, SurvivorRiskClass, "risk_class")


def is_high_risk(
    risk_class: SurvivorRiskClass | str,
    *,
    high_risk_classes: Sequence[str] | None = None,
) -> bool:
    """Return True when *risk_class* is in the high-risk set."""

    risk = _risk_value(risk_class)
    allowed = (
        tuple(high_risk_classes)
        if high_risk_classes is not None
        else DEFAULT_HIGH_RISK_CLASSES
    )
    sealed = tuple(_risk_value(item) for item in allowed)
    return risk in set(sealed)


def high_risk_classes() -> tuple[str, ...]:
    """Return the default high-risk class vocabulary."""

    return DEFAULT_HIGH_RISK_CLASSES


def _severity_for_risk(risk_class: str) -> str:
    return _RISK_TO_SEVERITY.get(risk_class, GapSeverity.MEDIUM.value)


def _structured_quality(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Project minimizer quality into DAG-JSON-safe structured values.

    ``MinimizationQuality.score`` is a float and is not admissible under
    structured identity; encode millipoints as an int and drop non-structured
    values.
    """

    if not value:
        return MappingProxyType({})
    out: dict[str, Any] = {}
    for key, item in dict(value).items():
        key_text = str(key)
        if type(item) is bool or item is None or type(item) is str:
            out[key_text] = item
        elif type(item) is int and not isinstance(item, bool):
            out[key_text] = item
        elif type(item) is float:
            # Preserve precision as integer millipoints (0..1000).
            out[key_text] = int(round(max(0.0, min(float(item), 1.0)) * 1000))
            out[f"{key_text}_encoding"] = "millipoints"
        elif isinstance(item, (list, tuple)):
            out[key_text] = [
                str(entry) if type(entry) is not int and type(entry) is not bool else entry
                for entry in item
            ]
        elif isinstance(item, Enum):
            out[key_text] = item.value
        else:
            out[key_text] = str(item)
    return MappingProxyType(out)


def _repository_state_cid(value: str | Mapping[str, Any]) -> str:
    if isinstance(value, Mapping):
        if "repository_state_cid" in value:
            return _cid(value["repository_state_cid"], "repository_state_cid")
        if "cid" in value:
            return _cid(value["cid"], "repository_state.cid")
        raise DiagnosisRuntimeError(
            "repository_state mapping requires repository_state_cid or cid"
        )
    return _cid(value, "repository_state")


def _normalize_mutation(
    value: DiagnosisMutationBinding | Mapping[str, Any],
) -> DiagnosisMutationBinding:
    if isinstance(value, DiagnosisMutationBinding):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value or "binding_cid" in value:
                return DiagnosisMutationBinding.from_dict(value)
            return DiagnosisMutationBinding(**dict(value))  # type: ignore[arg-type]
        except (SemanticDiagnosisError, TypeError, KeyError, AnalysisContractError) as exc:
            raise DiagnosisRuntimeError(
                f"mutation binding is malformed: {exc}"
            ) from exc
    raise DiagnosisRuntimeError(
        "mutation must be DiagnosisMutationBinding or mapping"
    )


def _normalize_outcome(
    value: DiagnosisOutcomeBinding | Mapping[str, Any],
) -> DiagnosisOutcomeBinding:
    if isinstance(value, DiagnosisOutcomeBinding):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value or "binding_cid" in value:
                return DiagnosisOutcomeBinding.from_dict(value)
            return DiagnosisOutcomeBinding(**dict(value))  # type: ignore[arg-type]
        except (SemanticDiagnosisError, TypeError, KeyError) as exc:
            raise DiagnosisRuntimeError(
                f"outcome binding is malformed: {exc}"
            ) from exc
    raise DiagnosisRuntimeError(
        "outcome must be DiagnosisOutcomeBinding or mapping"
    )


def _normalize_signals(
    value: DiagnosisSignals | Mapping[str, Any] | None,
) -> DiagnosisSignals:
    if value is None:
        return DiagnosisSignals(observation_complete=True)
    if isinstance(value, DiagnosisSignals):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value or "signals_cid" in value:
                return DiagnosisSignals.from_dict(value)
            return DiagnosisSignals(**dict(value))  # type: ignore[arg-type]
        except (SemanticDiagnosisError, TypeError, KeyError, AnalysisContractError) as exc:
            raise DiagnosisRuntimeError(
                f"diagnosis signals are malformed: {exc}"
            ) from exc
    raise DiagnosisRuntimeError("signals must be DiagnosisSignals or mapping")


def _normalize_comparison(
    value: DetectionComparisonResult | Mapping[str, Any] | None,
) -> DetectionComparisonResult | None:
    if value is None:
        return None
    if isinstance(value, DetectionComparisonResult):
        return value
    if isinstance(value, Mapping):
        try:
            return DetectionComparisonResult.from_dict(value)
        except (GapClassificationError, TypeError, KeyError) as exc:
            raise DiagnosisRuntimeError(
                f"comparison is malformed: {exc}"
            ) from exc
    raise DiagnosisRuntimeError(
        "comparison must be DetectionComparisonResult or mapping"
    )


def _normalize_header(
    value: AssuranceArtifactHeader | Mapping[str, Any],
    name: str = "header",
) -> AssuranceArtifactHeader:
    if isinstance(value, AssuranceArtifactHeader):
        return value
    if isinstance(value, Mapping):
        try:
            return AssuranceArtifactHeader.from_dict(value)
        except Exception as exc:
            raise DiagnosisRuntimeError(f"{name} is malformed: {exc}") from exc
    raise DiagnosisRuntimeError(
        f"{name} must be AssuranceArtifactHeader or mapping"
    )


def _normalize_source_spans(
    values: Sequence[SourceSpan | Mapping[str, Any]],
) -> tuple[SourceSpan, ...]:
    out: list[SourceSpan] = []
    for index, item in enumerate(values):
        if isinstance(item, SourceSpan):
            out.append(item)
            continue
        if isinstance(item, Mapping):
            try:
                if "schema" in item or "span_cid" in item:
                    out.append(SourceSpan.from_dict(item))
                else:
                    out.append(SourceSpan(**dict(item)))  # type: ignore[arg-type]
            except (AnalysisContractError, TypeError, KeyError) as exc:
                raise DiagnosisRuntimeError(
                    f"source_spans[{index}] is malformed: {exc}"
                ) from exc
            continue
        raise DiagnosisRuntimeError(
            f"source_spans[{index}] must be SourceSpan or mapping"
        )
    if not out:
        raise DiagnosisRuntimeError("source_spans must not be empty")
    return tuple(out)


def _artifact_header(
    base: AssuranceArtifactHeader,
    *,
    artifact_kind: str,
    interface_id: str,
    symbol_ids: Sequence[str] | None = None,
    repository_state_cid: str | None = None,
    receipt_cids: Sequence[str] | None = None,
    proof_cids: Sequence[str] | None = None,
) -> AssuranceArtifactHeader:
    generator = GeneratorIdentity(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        interface_id=interface_id,
    )
    versions = VersionBinding(
        operator_id=base.versions.operator_id,
        operator_version=base.versions.operator_version,
        campaign_policy_id=base.versions.campaign_policy_id,
        campaign_policy_version=base.versions.campaign_policy_version,
        generator=generator,
    )
    return AssuranceArtifactHeader(
        artifact_kind=artifact_kind,
        repository_id=base.repository_id,
        repository_state_cid=repository_state_cid or base.repository_state_cid,
        target_symbol_ids=(
            tuple(symbol_ids)
            if symbol_ids is not None
            else tuple(base.target_symbol_ids)
        ),
        target_artifact_cids=tuple(base.target_artifact_cids),
        capsule_cids=tuple(base.capsule_cids),
        proof_unit_cids=tuple(base.proof_unit_cids),
        environment_cid=base.environment_cid,
        dependency_lock_cid=base.dependency_lock_cid,
        versions=versions,
        provenance=base.provenance,
        terminal_status=base.terminal_status,
        receipt_cids=(
            tuple(receipt_cids)
            if receipt_cids is not None
            else tuple(base.receipt_cids)
        ),
        proof_cids=(
            tuple(proof_cids) if proof_cids is not None else tuple(base.proof_cids)
        ),
        metadata=dict(base.metadata),
    )


# ---------------------------------------------------------------------------
# Bounded reproduction / minimization orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundedReproduction:
    """Bounded reproduction evidence produced by minimization (or prebuilt)."""

    schema: str = BOUNDED_REPRODUCTION_SCHEMA
    minimization_status: str = MinimizationStatus.MINIMIZED.value
    minimization_failed: bool = False
    minimization_failure_reason: str | None = None
    evidence: MinimizedEvidenceBinding | None = None
    reproduction_command: str = ""
    reproduction_argv: tuple[str, ...] = ()
    reproduction_input_cid: str | None = None
    counterexample_receipt_cid: str | None = None
    failure_identity_cid: str | None = None
    guarantee: str = MinimizationGuarantee.NONE.value
    reason_codes: tuple[str, ...] = ()
    bounded_log_digest: BoundedLogDigest | None = None
    quality: Mapping[str, Any] = field(default_factory=dict)
    logs_bounded: bool = True
    used_counterexample_minimizer: bool = False

    def __post_init__(self) -> None:
        status = _enum(
            self.minimization_status, MinimizationStatus, "minimization_status"
        )
        object.__setattr__(self, "minimization_status", status)
        failed = _bool(self.minimization_failed, "minimization_failed")
        object.__setattr__(self, "minimization_failed", failed)
        if failed and status != MinimizationStatus.FAILED.value:
            raise DiagnosisRuntimeError(
                "minimization_failed requires minimization_status=failed"
            )
        if not failed and status != MinimizationStatus.MINIMIZED.value:
            raise DiagnosisRuntimeError(
                "successful reproduction requires minimization_status=minimized"
            )
        reason = _optional_text(
            self.minimization_failure_reason, "minimization_failure_reason"
        )
        if failed and reason is None:
            raise DiagnosisRuntimeError(
                "minimization_failure_reason is required when minimization fails"
            )
        if not failed and reason is not None:
            raise DiagnosisRuntimeError(
                "minimization_failure_reason requires failed status"
            )
        object.__setattr__(self, "minimization_failure_reason", reason)
        if self.evidence is not None and not isinstance(
            self.evidence, MinimizedEvidenceBinding
        ):
            raise DiagnosisRuntimeError(
                "evidence must be MinimizedEvidenceBinding"
            )
        if not failed and self.evidence is None:
            raise DiagnosisRuntimeError(
                "successful reproduction requires minimized evidence"
            )
        if self.evidence is not None:
            if self.evidence.minimization_failed != failed:
                raise DiagnosisRuntimeError(
                    "evidence.minimization_failed must match minimization_failed"
                )
        object.__setattr__(
            self,
            "reproduction_command",
            _text(self.reproduction_command, "reproduction_command", empty=True),
        )
        argv = tuple(
            _text(item, "reproduction_argv", empty=False)
            for item in (self.reproduction_argv or ())
        )
        if len(argv) > MAX_ARGV:
            raise DiagnosisRuntimeError("reproduction_argv exceeds maximum length")
        object.__setattr__(self, "reproduction_argv", argv)
        object.__setattr__(
            self,
            "reproduction_input_cid",
            _optional_cid(self.reproduction_input_cid, "reproduction_input_cid"),
        )
        object.__setattr__(
            self,
            "counterexample_receipt_cid",
            _optional_cid(
                self.counterexample_receipt_cid, "counterexample_receipt_cid"
            ),
        )
        object.__setattr__(
            self,
            "failure_identity_cid",
            _optional_cid(self.failure_identity_cid, "failure_identity_cid"),
        )
        try:
            guarantee = (
                self.guarantee
                if isinstance(self.guarantee, MinimizationGuarantee)
                else MinimizationGuarantee(str(self.guarantee))
            )
        except ValueError as exc:
            raise DiagnosisRuntimeError(
                f"unsupported minimization guarantee {self.guarantee!r}"
            ) from exc
        object.__setattr__(self, "guarantee", guarantee.value)
        object.__setattr__(
            self,
            "reason_codes",
            _stable_unique(
                [
                    code
                    for code in (self.reason_codes or ())
                    if isinstance(code, str) and code
                ]
            ),
        )
        if self.bounded_log_digest is not None and not isinstance(
            self.bounded_log_digest, BoundedLogDigest
        ):
            raise DiagnosisRuntimeError(
                "bounded_log_digest must be BoundedLogDigest"
            )
        if failed and self.bounded_log_digest is None:
            raise DiagnosisRuntimeError(
                "bounded_log_digest is required when minimization fails"
            )
        if not _bool(self.logs_bounded, "logs_bounded"):
            raise DiagnosisRuntimeError("logs_bounded must be true")
        object.__setattr__(self, "logs_bounded", True)
        object.__setattr__(
            self,
            "used_counterexample_minimizer",
            _bool(
                self.used_counterexample_minimizer,
                "used_counterexample_minimizer",
            ),
        )
        object.__setattr__(
            self, "quality", _structured_quality(dict(self.quality or {}))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "minimization_status": self.minimization_status,
            "minimization_failed": self.minimization_failed,
            "minimization_failure_reason": self.minimization_failure_reason,
            "evidence": (
                None if self.evidence is None else self.evidence.to_dict()
            ),
            "reproduction_command": self.reproduction_command,
            "reproduction_argv": list(self.reproduction_argv),
            "reproduction_input_cid": self.reproduction_input_cid,
            "counterexample_receipt_cid": self.counterexample_receipt_cid,
            "failure_identity_cid": self.failure_identity_cid,
            "guarantee": self.guarantee,
            "reason_codes": list(self.reason_codes),
            "bounded_log_digest": (
                None
                if self.bounded_log_digest is None
                else self.bounded_log_digest.to_dict()
            ),
            "quality": dict(self.quality),
            "logs_bounded": self.logs_bounded,
            "used_counterexample_minimizer": self.used_counterexample_minimizer,
        }


def _command_from_argv(argv: Sequence[str]) -> str:
    if not argv:
        return "pytest -q"
    # Shell-safe-ish join for durable report surface (already NFC text).
    return " ".join(argv)[:MAX_TEXT_CHARS]


def _bounded_digest_from_artifacts(
    artifact_cids: Sequence[str],
    *,
    reason: str,
    byte_count: int = DEFAULT_LOG_DIGEST_BYTES,
) -> BoundedLogDigest:
    seeds = list(artifact_cids) or ["bounded-log-empty"]
    digest_cid = cid_for_structured(
        {
            "kind": "bounded_log_digest_seed",
            "artifact_cids": list(seeds),
            "reason": reason,
        }
    )
    return BoundedLogDigest(
        digest_cid=digest_cid,
        byte_count=min(max(int(byte_count), 0), 1_048_576),
        truncated=True,
        full_log_excluded=True,
        notes=_clip(f"full logs excluded; {reason}", limit=512),
    )


def _evidence_from_counterexample(
    result: CounterexampleMinimizationResult,
) -> tuple[MinimizedEvidenceBinding, str | None, BoundedLogDigest | None, str]:
    """Map a CounterexampleMinimizationResult into AAE minimized evidence."""

    receipt = result.receipt
    quality = result.quality
    reasons = list(quality.reason_codes) + list(receipt.reason_codes)
    reasons = list(dict.fromkeys(str(item) for item in reasons if str(item).strip()))

    artifact_cids = list(receipt.artifact_cids or ())
    if result.failure_identity_cid:
        artifact_cids.append(result.failure_identity_cid)
    if receipt.failed_receipt_cid:
        artifact_cids.append(receipt.failed_receipt_cid)
    # Stable unique CIDs.
    unique_cids: list[str] = []
    seen: set[str] = set()
    for item in artifact_cids:
        try:
            sealed = validate_cid(item)
        except Exception:
            continue
        if sealed in seen:
            continue
        seen.add(sealed)
        unique_cids.append(sealed)

    if receipt.minimized:
        if not unique_cids:
            unique_cids = [result.failure_identity_cid]
        repro_input = result.failure_identity_cid
        evidence = MinimizedEvidenceBinding(
            evidence_cids=tuple(unique_cids),
            minimized=True,
            minimization_failed=False,
            reproduction_input_cid=repro_input,
            notes="counterexample minimizer produced bounded reproduction",
        )
        return evidence, None, None, REASON_MINIMIZATION_SUCCEEDED

    reason = (
        "; ".join(reasons)
        if reasons
        else "counterexample minimization failed without lease-validated reduction"
    )
    digest = _bounded_digest_from_artifacts(
        unique_cids or [result.failure_identity_cid],
        reason=reason,
    )
    # On failure, evidence may be empty; failure is carried by flags + digest.
    evidence = MinimizedEvidenceBinding(
        evidence_cids=tuple(unique_cids) if unique_cids else (),
        minimized=False,
        minimization_failed=True,
        reproduction_input_cid=None,
        notes=_clip(f"minimization_failed: {reason}", limit=MAX_TEXT_CHARS),
    )
    return evidence, reason, digest, REASON_MINIMIZATION_FAILED


def run_counterexample_minimization(
    *,
    request: MinimizationRequest | None = None,
    failed_receipt: Any | None = None,
    failure_material: FailureMaterial | None = None,
    reproduction_argv: Sequence[str] | None = None,
    rerun_oracle: RerunOracle | None = None,
    process_runner: Any | None = None,
    command_template: Any | None = None,
    budget: MinimizationBudget | None = None,
    semantic_cone_paths: Sequence[str] = (),
    semantic_cone_symbols: Sequence[str] = (),
    minimizer: CounterexampleMinimizer | None = None,
) -> tuple[BoundedReproduction, CounterexampleMinimizationResult]:
    """Run ``CounterexampleMinimizer@1`` and map to a bounded reproduction."""

    try:
        if request is not None:
            engine = minimizer or CounterexampleMinimizer()
            result = engine.minimize(request)
        elif failed_receipt is not None:
            result = minimize_counterexample(
                failed_receipt,
                failure_material,
                reproduction_argv=reproduction_argv,
                process_runner=process_runner,
                command_template=command_template,
                rerun_oracle=rerun_oracle,
                budget=budget,
                semantic_cone_paths=semantic_cone_paths,
                semantic_cone_symbols=semantic_cone_symbols,
            )
        else:
            raise DiagnosisRuntimeError(
                "counterexample minimization requires MinimizationRequest or "
                "failed_receipt",
                reason_code="missing_minimization_input",
            )
    except CounterexampleMinimizationError as exc:
        raise DiagnosisRuntimeError(
            f"counterexample minimization failed closed: {exc}",
            reason_code="counterexample_minimization_error",
        ) from exc

    evidence, failure_reason, digest, primary_reason = _evidence_from_counterexample(
        result
    )
    argv = tuple(result.accepted_argv) or tuple(
        result.receipt.reproduction_argv or ()
    )
    raw_quality = (
        result.quality.to_dict()
        if isinstance(result.quality, MinimizationQuality)
        else {}
    )
    quality = dict(_structured_quality(raw_quality))
    reproduction = BoundedReproduction(
        minimization_status=(
            MinimizationStatus.FAILED.value
            if evidence.minimization_failed
            else MinimizationStatus.MINIMIZED.value
        ),
        minimization_failed=evidence.minimization_failed,
        minimization_failure_reason=failure_reason,
        evidence=evidence,
        reproduction_command=_command_from_argv(argv),
        reproduction_argv=argv,
        reproduction_input_cid=evidence.reproduction_input_cid,
        counterexample_receipt_cid=getattr(result.receipt, "receipt_id", None)
        or result.failure_identity_cid,
        failure_identity_cid=result.failure_identity_cid,
        guarantee=result.quality.guarantee.value
        if isinstance(result.quality.guarantee, MinimizationGuarantee)
        else str(result.quality.guarantee),
        reason_codes=_stable_unique(
            [
                REASON_COUNTEREXAMPLE_MINIMIZER,
                primary_reason,
                REASON_BOUNDED_REPRODUCTION,
                REASON_LOGS_BOUNDED,
                *list(result.quality.reason_codes),
            ]
        ),
        bounded_log_digest=digest,
        quality=quality,
        logs_bounded=True,
        used_counterexample_minimizer=True,
    )
    return reproduction, result


def _reproduction_from_prebuilt(
    *,
    evidence: MinimizedEvidenceBinding | Mapping[str, Any] | None,
    minimization_status: MinimizationStatus | str | None = None,
    minimization_failure_reason: str | None = None,
    bounded_log_digest: BoundedLogDigest | Mapping[str, Any] | None = None,
    reproduction_command: str | None = None,
    reproduction_argv: Sequence[str] = (),
    reproduction_input_cid: str | None = None,
) -> BoundedReproduction:
    """Build a BoundedReproduction from pre-minimized subject evidence."""

    sealed_evidence: MinimizedEvidenceBinding | None = None
    if evidence is not None:
        if isinstance(evidence, MinimizedEvidenceBinding):
            sealed_evidence = evidence
        elif isinstance(evidence, Mapping):
            try:
                if "schema" in evidence or "binding_cid" in evidence:
                    sealed_evidence = MinimizedEvidenceBinding.from_dict(evidence)
                else:
                    sealed_evidence = MinimizedEvidenceBinding(
                        **dict(evidence)  # type: ignore[arg-type]
                    )
            except (AnalysisContractError, TypeError, KeyError) as exc:
                raise DiagnosisRuntimeError(
                    f"minimized_evidence is malformed: {exc}"
                ) from exc
        else:
            raise DiagnosisRuntimeError(
                "minimized_evidence must be MinimizedEvidenceBinding or mapping"
            )

    failed = False
    status = MinimizationStatus.MINIMIZED.value
    reason = minimization_failure_reason
    digest: BoundedLogDigest | None = None

    if sealed_evidence is not None:
        failed = sealed_evidence.minimization_failed
        status = (
            MinimizationStatus.FAILED.value
            if failed
            else MinimizationStatus.MINIMIZED.value
        )
    if minimization_status is not None:
        status = _enum(
            minimization_status, MinimizationStatus, "minimization_status"
        )
        failed = status == MinimizationStatus.FAILED.value

    if bounded_log_digest is not None:
        if isinstance(bounded_log_digest, BoundedLogDigest):
            digest = bounded_log_digest
        elif isinstance(bounded_log_digest, Mapping):
            try:
                if "schema" in bounded_log_digest or "digest_binding_cid" in bounded_log_digest:
                    digest = BoundedLogDigest.from_dict(bounded_log_digest)
                else:
                    digest = BoundedLogDigest(**dict(bounded_log_digest))  # type: ignore[arg-type]
            except (SurvivorMinimizationError, TypeError, KeyError) as exc:
                raise DiagnosisRuntimeError(
                    f"bounded_log_digest is malformed: {exc}"
                ) from exc
        else:
            raise DiagnosisRuntimeError(
                "bounded_log_digest must be BoundedLogDigest or mapping"
            )

    if failed:
        if reason is None:
            reason = (
                sealed_evidence.notes
                if sealed_evidence is not None and sealed_evidence.notes
                else "bounded minimization failed"
            )
        if digest is None:
            seeds = (
                list(sealed_evidence.evidence_cids)
                if sealed_evidence is not None
                else []
            )
            digest = _bounded_digest_from_artifacts(seeds, reason=reason)
        if sealed_evidence is None:
            sealed_evidence = MinimizedEvidenceBinding(
                evidence_cids=(),
                minimized=False,
                minimization_failed=True,
                reproduction_input_cid=None,
                notes=_clip(f"minimization_failed: {reason}", limit=MAX_TEXT_CHARS),
            )
    else:
        if sealed_evidence is None:
            raise DiagnosisRuntimeError(
                "successful prebuilt reproduction requires minimized_evidence"
            )
        reason = None

    command = reproduction_command or "pytest -q"
    repro_input = reproduction_input_cid
    if repro_input is None and sealed_evidence is not None:
        repro_input = sealed_evidence.reproduction_input_cid

    return BoundedReproduction(
        minimization_status=status,
        minimization_failed=failed,
        minimization_failure_reason=reason,
        evidence=sealed_evidence,
        reproduction_command=command,
        reproduction_argv=tuple(reproduction_argv or ()),
        reproduction_input_cid=repro_input,
        guarantee=(
            MinimizationGuarantee.NONE.value
            if failed
            else MinimizationGuarantee.BOUNDED.value
        ),
        reason_codes=_stable_unique(
            [
                REASON_MINIMIZATION_SKIPPED,
                REASON_MINIMIZATION_FAILED if failed else REASON_MINIMIZATION_SUCCEEDED,
                REASON_BOUNDED_REPRODUCTION,
                REASON_LOGS_BOUNDED,
            ]
        ),
        bounded_log_digest=digest,
        logs_bounded=True,
        used_counterexample_minimizer=False,
    )


# ---------------------------------------------------------------------------
# Survivor report construction
# ---------------------------------------------------------------------------


def _build_survivor_report(
    *,
    mutation: DiagnosisMutationBinding,
    outcome: DiagnosisOutcomeBinding,
    repository_state_cid: str,
    reproduction: BoundedReproduction,
    detectors_run: Sequence[str],
    detectors_omitted: Sequence[str],
    expected_behavior: str,
    observed_behavior: str,
    proof_cids: Sequence[str] = (),
    receipt_cids: Sequence[str] = (),
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SurvivorReportBuildResult:
    assert reproduction.evidence is not None
    evidence = reproduction.evidence
    subject = SurvivorMinimizationSubject(
        subject_id=f"subj.{mutation.candidate_id}",
        report_id=f"survivor_{mutation.candidate_id}",
        candidate_id=mutation.candidate_id,
        candidate_cid=mutation.candidate_cid,
        outcome_cid=outcome.outcome_cid,
        risk_class=mutation.risk_class,
        symbol_ids=mutation.symbol_ids,
        violated_or_missing_property=mutation.violated_or_missing_property,
        source_spans=mutation.source_spans,
        detectors_run=detectors_run,
        detectors_omitted=detectors_omitted,
        expected_behavior=expected_behavior,
        observed_behavior=observed_behavior,
        dependency_path=mutation.dependency_path,
        reproduction_command=reproduction.reproduction_command
        or "pytest -q",
        evidence_cids=evidence.evidence_cids,
        reproduction_input_cid=evidence.reproduction_input_cid,
        proof_cids=proof_cids or tuple(mutation.header.proof_cids),
        receipt_cids=receipt_cids or tuple(mutation.header.receipt_cids),
        equivalence_assessment_cid=outcome.equivalence_assessment_cid,
        minimization_status=reproduction.minimization_status,
        minimization_failure_reason=reproduction.minimization_failure_reason,
        bounded_log_digest=reproduction.bounded_log_digest,
        observation_complete=True,
        repository_state_cid=repository_state_cid,
        notes=notes,
        metadata={
            **dict(metadata or {}),
            "diagnosis_runtime": GENERATOR_ID,
            "used_counterexample_minimizer": (
                reproduction.used_counterexample_minimizer
            ),
            "minimizer_interface": COUNTEREXAMPLE_MINIMIZER_INTERFACE
            if reproduction.used_counterexample_minimizer
            else None,
        },
    )
    header = _artifact_header(
        mutation.header,
        artifact_kind="surviving_mutant_report",
        interface_id=BUILD_SURVIVING_MUTANT_REPORT_INTERFACE,
        symbol_ids=mutation.symbol_ids,
        repository_state_cid=repository_state_cid,
    )
    try:
        result = build_surviving_mutant_report(subject, header, notes=notes)
    except SurvivorMinimizationError as exc:
        raise DiagnosisRuntimeError(
            f"survivor report build failed: {exc}",
            reason_code="survivor_report_failed",
        ) from exc
    if not result.logs_bounded or not logs_remain_bounded(result):
        raise DiagnosisRuntimeError(
            "survivor report must keep logs bounded",
            reason_code="logs_not_bounded",
        )
    if result.minimization_failed != reproduction.minimization_failed:
        raise DiagnosisRuntimeError(
            "survivor report minimization_failed flag diverged from reproduction",
            reason_code="minimization_flag_mismatch",
        )
    return result


# ---------------------------------------------------------------------------
# Assurance gap construction and persistence
# ---------------------------------------------------------------------------


def _gap_class_for_diagnosis(diagnosis: SurvivorDiagnosis) -> str:
    if diagnosis.gap_class is not None:
        return str(diagnosis.gap_class)
    disposition = str(diagnosis.disposition)
    if disposition == DiagnosisDisposition.EQUIVALENT.value:
        # Equivalence is recorded; still a sealed gap for high-risk survivors.
        return AssuranceGapClass.PROBABLY_EQUIVALENT.value
    if disposition == DiagnosisDisposition.PROBABLY_EQUIVALENT.value:
        return AssuranceGapClass.PROBABLY_EQUIVALENT.value
    if disposition == DiagnosisDisposition.INTENTIONALLY_UNCONSTRAINED.value:
        return AssuranceGapClass.INTENTIONALLY_UNCONSTRAINED.value
    if disposition == DiagnosisDisposition.SPECIFICATION_AMBIGUITY.value:
        return AssuranceGapClass.SPECIFICATION_AMBIGUITY.value
    if disposition == DiagnosisDisposition.PRODUCT_DEFECT.value:
        return AssuranceGapClass.UNKNOWN.value
    return AssuranceGapClass.UNKNOWN.value


def _build_assurance_gap(
    *,
    mutation: DiagnosisMutationBinding,
    diagnosis: SurvivorDiagnosis,
    reproduction: BoundedReproduction,
    survivor_report_cid: str | None,
    comparison: DetectionComparisonResult | None,
    high_risk: bool,
) -> AssuranceGap:
    """Seal an AssuranceGap for a survivor.

    Unknown gaps always require human review. Human review accompanies the
    gap — it never replaces gap persistence for high-risk survivors.
    """

    assert reproduction.evidence is not None
    gap_class = _gap_class_for_diagnosis(diagnosis)
    requires_review = bool(diagnosis.requires_human_review)
    if gap_class == AssuranceGapClass.UNKNOWN.value:
        requires_review = True

    severity = (
        str(diagnosis.severity)
        if diagnosis.severity is not None
        else _severity_for_risk(str(mutation.risk_class))
    )

    # Prefer diagnosis-derived gap class when present. classify_assurance_gap
    # without a comparison cannot see detector-partition signals and would
    # incorrectly collapse to unknown.
    equivalence_status: str | None = None
    intentionally = False
    ambiguous = False
    disposition = str(diagnosis.disposition)
    if disposition == DiagnosisDisposition.EQUIVALENT.value:
        equivalence_status = EquivalenceAssessmentStatus.EQUIVALENT.value
    elif disposition == DiagnosisDisposition.PROBABLY_EQUIVALENT.value:
        equivalence_status = EquivalenceAssessmentStatus.PROBABLY_EQUIVALENT.value
    elif disposition == DiagnosisDisposition.INTENTIONALLY_UNCONSTRAINED.value:
        intentionally = True
    elif disposition == DiagnosisDisposition.SPECIFICATION_AMBIGUITY.value:
        ambiguous = True

    gap_header = _artifact_header(
        mutation.header,
        artifact_kind="assurance_gap",
        interface_id=CLASSIFY_ASSURANCE_GAP_INTERFACE,
        symbol_ids=mutation.symbol_ids,
        repository_state_cid=diagnosis.repository_state_cid,
    )
    gap: AssuranceGap | None = None

    # Use classify_assurance_gap when comparison is available or diagnosis has
    # no concrete gap class (equivalence / intentional / residual unknown).
    use_classifier = comparison is not None or diagnosis.gap_class is None
    if use_classifier:
        subject = GapClassificationSubject(
            candidate_id=mutation.candidate_id,
            candidate_cid=mutation.candidate_cid,
            risk_class=mutation.risk_class,
            violated_or_missing_property=mutation.violated_or_missing_property,
            symbol_ids=mutation.symbol_ids,
            source_spans=mutation.source_spans,
            dependency_path=mutation.dependency_path,
            minimized_evidence=reproduction.evidence,
            header=gap_header,
            gap_id=f"gap.{mutation.candidate_id}",
            survivor_report_cid=survivor_report_cid,
            equivalence_status=equivalence_status,
            intentionally_unconstrained=intentionally,
            specification_ambiguous=ambiguous,
            observation_complete=True,
            notes=diagnosis.notes,
            metadata={
                "diagnosis_cid": diagnosis.diagnosis_cid,
                "diagnosis_disposition": disposition,
                "high_risk": high_risk,
                "human_review_accompanies_gap": True,
            },
        )
        try:
            classified = classify_assurance_gap(subject, comparison)
            # When diagnosis already decided a concrete detector-derived class
            # and the classifier only saw residual unknown (no comparison),
            # prefer the diagnosis class.
            if (
                diagnosis.gap_class is not None
                and comparison is None
                and classified.gap_class == AssuranceGapClass.UNKNOWN.value
                and str(diagnosis.gap_class) != AssuranceGapClass.UNKNOWN.value
            ):
                gap = None
            else:
                gap = classified
        except GapClassificationError:
            gap = None

    if gap is None:
        # Seal from diagnosis fields so high-risk never loses its AssuranceGap
        # and detector-derived classes survive without a comparison envelope.
        gap = AssuranceGap(
            header=gap_header,
            gap_id=f"gap.{mutation.candidate_id}",
            gap_class=gap_class,
            severity=severity,
            risk_class=mutation.risk_class,
            summary=_clip(
                diagnosis.summary
                or f"assurance gap for high-risk survivor {mutation.candidate_id}",
                limit=MAX_TEXT_CHARS,
            ),
            candidate_id=mutation.candidate_id,
            candidate_cid=mutation.candidate_cid,
            survivor_report_cid=survivor_report_cid,
            violated_or_missing_property=mutation.violated_or_missing_property,
            symbol_ids=mutation.symbol_ids,
            source_spans=mutation.source_spans,
            dependency_path=mutation.dependency_path,
            minimized_evidence=reproduction.evidence,
            requires_human_review=requires_review,
            detection_failure_cids=(),
            vacuity_finding_cids=(),
            notes=diagnosis.notes,
            metadata={
                "diagnosis_cid": diagnosis.diagnosis_cid,
                "diagnosis_disposition": disposition,
                "high_risk": high_risk,
                "human_review_accompanies_gap": True,
                "gap_source": "survivor_diagnosis",
            },
        )

    # Enforce: unknown always requires human review; human review never drops
    # the gap (gap object remains the authority for high-risk persistence).
    if gap.gap_class == AssuranceGapClass.UNKNOWN.value and not gap.requires_human_review:
        gap = AssuranceGap(
            header=gap.header,
            gap_id=gap.gap_id,
            gap_class=gap.gap_class,
            severity=gap.severity,
            risk_class=gap.risk_class,
            summary=gap.summary,
            candidate_id=gap.candidate_id,
            candidate_cid=gap.candidate_cid,
            survivor_report_cid=gap.survivor_report_cid,
            violated_or_missing_property=gap.violated_or_missing_property,
            symbol_ids=gap.symbol_ids,
            source_spans=gap.source_spans,
            dependency_path=gap.dependency_path,
            minimized_evidence=gap.minimized_evidence,
            requires_human_review=True,
            detection_failure_cids=gap.detection_failure_cids,
            vacuity_finding_cids=gap.vacuity_finding_cids,
            notes=gap.notes,
            metadata={
                **dict(gap.metadata),
                "human_review_forced_for_unknown": True,
            },
        )
    verify_gap_identity(gap)
    return gap


def _persist_gap(
    gap: AssuranceGap,
    *,
    repository: AssuranceGapRepository | None,
    request: GapPersistRequest | None,
) -> tuple[str, GapPersistStatus, Mapping[str, Any] | None]:
    """Persist gap when repository is provided; otherwise seal-only."""

    if repository is None:
        return gap.gap_cid, GapPersistStatus.SEALED_ONLY, None
    persist = request or GapPersistRequest()
    try:
        result = repository.persist_gap(
            persist.workspace,
            gap.to_dict(),
            expected_cid=gap.gap_cid,
            artifact_operation_id=persist.artifact_operation_id,
            history_operation_id=persist.history_operation_id,
            expected_history_generation=persist.expected_history_generation,
            expected_history_head_cid=persist.expected_history_head_cid,
            replicate=persist.replicate,
        )
    except Exception as exc:
        raise DiagnosisRuntimeError(
            f"assurance gap persistence failed: {exc}",
            reason_code="gap_persist_failed",
        ) from exc
    result_map: Mapping[str, Any] | None
    if isinstance(result, Mapping):
        result_map = MappingProxyType(dict(result))
        result_cid = str(result.get("gap_cid") or gap.gap_cid)
    else:
        result_cid = str(getattr(result, "gap_cid", gap.gap_cid))
        result_map = MappingProxyType({"gap_cid": result_cid})
    if result_cid != gap.gap_cid:
        raise DiagnosisRuntimeError(
            f"persisted gap cid mismatch: {result_cid!r} != {gap.gap_cid!r}",
            reason_code="gap_persist_cid_mismatch",
        )
    return result_cid, GapPersistStatus.PERSISTED, result_map


# ---------------------------------------------------------------------------
# Sealed diagnosis run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SurvivorDiagnosisRun:
    """Sealed orchestration result for ``diagnose_surviving_mutant@1``."""

    schema: str = SURVIVOR_DIAGNOSIS_RUN_SCHEMA
    interface_id: str = SURVIVOR_DIAGNOSIS_RUN_INTERFACE
    run_cid: str = ""
    candidate_id: str = ""
    candidate_cid: str = ""
    outcome_cid: str = ""
    repository_state_cid: str = ""
    risk_class: str = SurvivorRiskClass.MEDIUM.value
    high_risk: bool = False
    phases: tuple[str, ...] = ()
    reproduction: BoundedReproduction | None = None
    survivor_report: SurvivingMutantReport | None = None
    survivor_report_cid: str | None = None
    diagnosis: SurvivorDiagnosis | None = None
    diagnosis_cid: str | None = None
    assurance_gap: AssuranceGap | None = None
    gap_cid: str | None = None
    gap_persist_status: str = GapPersistStatus.NOT_REQUIRED.value
    gap_persist_result: Mapping[str, Any] | None = None
    requires_human_review: bool = False
    human_review_accompanies_gap: bool = False
    minimization_failed: bool = False
    logs_bounded: bool = True
    production_policy_changed: bool = False
    reason_codes: tuple[str, ...] = ()
    evidence_subset: str = AAE_DIAGNOSIS_RUN_EVIDENCE
    diagnostic: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self, "interface_id", _text(self.interface_id, "interface_id")
        )
        object.__setattr__(
            self, "candidate_id", _token(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self, "outcome_cid", _cid(self.outcome_cid, "outcome_cid")
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            _cid(self.repository_state_cid, "repository_state_cid"),
        )
        object.__setattr__(self, "risk_class", _risk_value(self.risk_class))
        object.__setattr__(self, "high_risk", _bool(self.high_risk, "high_risk"))
        phases = tuple(
            _enum(phase, DiagnosisPhase, "phases") for phase in (self.phases or ())
        )
        object.__setattr__(self, "phases", phases)
        if self.reproduction is not None and not isinstance(
            self.reproduction, BoundedReproduction
        ):
            raise DiagnosisRuntimeError("reproduction must be BoundedReproduction")
        if self.survivor_report is not None and not isinstance(
            self.survivor_report, SurvivingMutantReport
        ):
            raise DiagnosisRuntimeError(
                "survivor_report must be SurvivingMutantReport"
            )
        object.__setattr__(
            self,
            "survivor_report_cid",
            _optional_cid(self.survivor_report_cid, "survivor_report_cid"),
        )
        if self.diagnosis is not None and not isinstance(
            self.diagnosis, SurvivorDiagnosis
        ):
            raise DiagnosisRuntimeError("diagnosis must be SurvivorDiagnosis")
        object.__setattr__(
            self,
            "diagnosis_cid",
            _optional_cid(self.diagnosis_cid, "diagnosis_cid"),
        )
        if self.assurance_gap is not None and not isinstance(
            self.assurance_gap, AssuranceGap
        ):
            raise DiagnosisRuntimeError("assurance_gap must be AssuranceGap")
        object.__setattr__(
            self, "gap_cid", _optional_cid(self.gap_cid, "gap_cid")
        )
        object.__setattr__(
            self,
            "gap_persist_status",
            _enum(self.gap_persist_status, GapPersistStatus, "gap_persist_status"),
        )
        if self.gap_persist_result is not None:
            object.__setattr__(
                self,
                "gap_persist_result",
                _mapping(self.gap_persist_result, "gap_persist_result"),
            )
        object.__setattr__(
            self,
            "requires_human_review",
            _bool(self.requires_human_review, "requires_human_review"),
        )
        object.__setattr__(
            self,
            "human_review_accompanies_gap",
            _bool(
                self.human_review_accompanies_gap,
                "human_review_accompanies_gap",
            ),
        )
        object.__setattr__(
            self,
            "minimization_failed",
            _bool(self.minimization_failed, "minimization_failed"),
        )
        if not _bool(self.logs_bounded, "logs_bounded"):
            raise DiagnosisRuntimeError("logs_bounded must be true")
        object.__setattr__(self, "logs_bounded", True)
        # Hard invariant: never claim production policy change.
        object.__setattr__(self, "production_policy_changed", False)
        object.__setattr__(
            self,
            "reason_codes",
            _stable_unique(
                [
                    code
                    for code in (self.reason_codes or ())
                    if isinstance(code, str) and code
                ]
            ),
        )
        object.__setattr__(
            self,
            "evidence_subset",
            _text(self.evidence_subset, "evidence_subset"),
        )
        object.__setattr__(
            self,
            "diagnostic",
            _clip(
                _text(self.diagnostic, "diagnostic", empty=True),
                limit=MAX_DIAGNOSTIC,
            )
            if self.diagnostic
            else "",
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        # High-risk survivors always bind an AssuranceGap.
        if self.high_risk and self.assurance_gap is None:
            raise HighRiskGapMissingError(
                "high-risk survivors must always persist an AssuranceGap",
                reason_code="high_risk_gap_missing",
            )
        if self.high_risk and self.gap_cid is None:
            raise HighRiskGapMissingError(
                "high-risk survivors must bind gap_cid",
                reason_code="high_risk_gap_cid_missing",
            )
        # Unknown gap requires human review alongside the gap (never instead).
        if (
            self.assurance_gap is not None
            and self.assurance_gap.gap_class == AssuranceGapClass.UNKNOWN.value
        ):
            if not self.assurance_gap.requires_human_review:
                raise DiagnosisRuntimeError(
                    "unknown AssuranceGap requires requires_human_review=true",
                    reason_code="unknown_gap_without_review",
                )
            if not self.requires_human_review:
                raise DiagnosisRuntimeError(
                    "unknown gap requires run-level human review accompaniment",
                    reason_code="unknown_gap_without_run_review",
                )
            if not self.human_review_accompanies_gap:
                raise DiagnosisRuntimeError(
                    "human review must accompany unknown gap, not replace it",
                    reason_code="human_review_replaced_gap",
                )

        if not self.run_cid:
            object.__setattr__(self, "run_cid", self.compute_run_cid())

    def compute_run_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "candidate_id": self.candidate_id,
            "candidate_cid": self.candidate_cid,
            "outcome_cid": self.outcome_cid,
            "repository_state_cid": self.repository_state_cid,
            "risk_class": self.risk_class,
            "high_risk": self.high_risk,
            "phases": list(self.phases),
            "reproduction": (
                None if self.reproduction is None else self.reproduction.to_dict()
            ),
            "survivor_report_cid": self.survivor_report_cid,
            "diagnosis_cid": self.diagnosis_cid,
            "gap_cid": self.gap_cid,
            "gap_persist_status": self.gap_persist_status,
            "requires_human_review": self.requires_human_review,
            "human_review_accompanies_gap": self.human_review_accompanies_gap,
            "minimization_failed": self.minimization_failed,
            "logs_bounded": self.logs_bounded,
            "production_policy_changed": False,
            "reason_codes": list(self.reason_codes),
            "evidence_subset": self.evidence_subset,
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["run_cid"] = self.run_cid
        payload["survivor_report"] = (
            None if self.survivor_report is None else self.survivor_report.to_dict()
        )
        payload["diagnosis"] = (
            None if self.diagnosis is None else self.diagnosis.to_dict()
        )
        payload["assurance_gap"] = (
            None if self.assurance_gap is None else self.assurance_gap.to_dict()
        )
        payload["gap_persist_result"] = (
            None
            if self.gap_persist_result is None
            else dict(self.gap_persist_result)
        )
        return payload


# ---------------------------------------------------------------------------
# Public: diagnose_surviving_mutant
# ---------------------------------------------------------------------------


def diagnose_surviving_mutant(
    mutation: DiagnosisMutationBinding | Mapping[str, Any],
    outcome: DiagnosisOutcomeBinding | Mapping[str, Any],
    repository_state: str | Mapping[str, Any],
    *,
    signals: DiagnosisSignals | Mapping[str, Any] | None = None,
    comparison: DetectionComparisonResult | Mapping[str, Any] | None = None,
    # Prebuilt or live minimization
    minimization_subject: SurvivorMinimizationSubject | Mapping[str, Any] | None = None,
    minimized_evidence: MinimizedEvidenceBinding | Mapping[str, Any] | None = None,
    minimization_request: MinimizationRequest | None = None,
    failed_receipt: Any | None = None,
    failure_material: FailureMaterial | None = None,
    reproduction_argv: Sequence[str] | None = None,
    rerun_oracle: RerunOracle | None = None,
    process_runner: Any | None = None,
    command_template: Any | None = None,
    minimization_budget: MinimizationBudget | None = None,
    semantic_cone_paths: Sequence[str] = (),
    semantic_cone_symbols: Sequence[str] = (),
    counterexample_minimizer: CounterexampleMinimizer | None = None,
    # Report surface extras
    detectors_run: Sequence[str] = (),
    detectors_omitted: Sequence[str] = (),
    expected_behavior: str | None = None,
    observed_behavior: str | None = None,
    reproduction_command: str | None = None,
    survivor_report: SurvivingMutantReport | Mapping[str, Any] | None = None,
    # Gap persistence
    gap_repository: AssuranceGapRepository | None = None,
    gap_persist_request: GapPersistRequest | Mapping[str, Any] | None = None,
    high_risk_classes: Sequence[str] | None = None,
    always_persist_gap: bool = False,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SurvivorDiagnosisRun:
    """Orchestrate minimization, semantic diagnosis, and high-risk gap persistence.

    Interface: ``diagnose_surviving_mutant@1``

    Plan signature: ``diagnose_surviving_mutant(mutation, outcome,
    repository_state)``.

    Pipeline:

    1. Admit mutation/outcome/repository bindings.
    2. Minimize via ``CounterexampleMinimizer`` or accept prebuilt bounded
       evidence; failure is always explicit.
    3. Build a bounded ``SurvivingMutantReport`` (unless provided).
    4. Run semantic nine-step diagnosis (datasets AAE-030).
    5. For every high-risk survivor, seal and optionally durable-persist an
       ``AssuranceGap``. Unknown gaps always require human review, which
       accompanies the gap rather than replacing it.
    """

    phases: list[str] = [DiagnosisPhase.ADMIT.value]
    reasons: list[str] = [REASON_NO_PRODUCTION_POLICY_CHANGE]

    sealed_mutation = _normalize_mutation(mutation)
    sealed_outcome = _normalize_outcome(outcome)
    repo_state_cid = _repository_state_cid(repository_state)
    sealed_signals = _normalize_signals(signals)
    sealed_comparison = _normalize_comparison(comparison)

    if sealed_mutation.candidate_id != sealed_outcome.candidate_id:
        raise DiagnosisRuntimeError(
            "mutation.candidate_id must match outcome.candidate_id"
        )
    if sealed_mutation.candidate_cid != sealed_outcome.candidate_cid:
        raise DiagnosisRuntimeError(
            "mutation.candidate_cid must match outcome.candidate_cid"
        )
    if not sealed_signals.observation_complete:
        raise DiagnosisRuntimeError(
            "diagnose_surviving_mutant fails closed when observation_complete is false",
            reason_code="incomplete_observation",
        )

    risk = _risk_value(sealed_mutation.risk_class)
    high_risk = is_high_risk(risk, high_risk_classes=high_risk_classes)

    # --- phase: minimize ---
    phases.append(DiagnosisPhase.MINIMIZE.value)
    counterexample_result: CounterexampleMinimizationResult | None = None
    reproduction: BoundedReproduction

    use_minimizer = (
        minimization_request is not None or failed_receipt is not None
    )
    if use_minimizer:
        reproduction, counterexample_result = run_counterexample_minimization(
            request=minimization_request,
            failed_receipt=failed_receipt,
            failure_material=failure_material,
            reproduction_argv=reproduction_argv,
            rerun_oracle=rerun_oracle,
            process_runner=process_runner,
            command_template=command_template,
            budget=minimization_budget,
            semantic_cone_paths=semantic_cone_paths,
            semantic_cone_symbols=semantic_cone_symbols,
            minimizer=counterexample_minimizer,
        )
        reasons.extend(reproduction.reason_codes)
    elif minimization_subject is not None:
        if isinstance(minimization_subject, SurvivorMinimizationSubject):
            subject = minimization_subject
        elif isinstance(minimization_subject, Mapping):
            try:
                if (
                    "schema" in minimization_subject
                    or "subject_observation_cid" in minimization_subject
                ):
                    subject = SurvivorMinimizationSubject.from_dict(
                        minimization_subject
                    )
                else:
                    subject = SurvivorMinimizationSubject(
                        **dict(minimization_subject)  # type: ignore[arg-type]
                    )
            except (SurvivorMinimizationError, TypeError, KeyError) as exc:
                raise DiagnosisRuntimeError(
                    f"minimization_subject is malformed: {exc}"
                ) from exc
        else:
            raise DiagnosisRuntimeError(
                "minimization_subject must be SurvivorMinimizationSubject or mapping"
            )
        reproduction = _reproduction_from_prebuilt(
            evidence=MinimizedEvidenceBinding(
                evidence_cids=subject.evidence_cids,
                minimized=subject.minimization_status
                == MinimizationStatus.MINIMIZED.value,
                minimization_failed=subject.minimization_status
                == MinimizationStatus.FAILED.value,
                reproduction_input_cid=subject.reproduction_input_cid,
                notes=subject.minimization_failure_reason or subject.notes,
            )
            if subject.minimization_status == MinimizationStatus.MINIMIZED.value
            or subject.evidence_cids
            or subject.minimization_status == MinimizationStatus.FAILED.value
            else MinimizedEvidenceBinding(
                evidence_cids=(),
                minimized=False,
                minimization_failed=True,
                notes=subject.minimization_failure_reason or "minimization failed",
            ),
            minimization_status=subject.minimization_status,
            minimization_failure_reason=subject.minimization_failure_reason,
            bounded_log_digest=subject.bounded_log_digest,
            reproduction_command=subject.reproduction_command,
            reproduction_input_cid=subject.reproduction_input_cid,
        )
        # Prefer subject detectors when caller did not override.
        if not detectors_run:
            detectors_run = subject.detectors_run
        if not detectors_omitted:
            detectors_omitted = subject.detectors_omitted
        if expected_behavior is None:
            expected_behavior = subject.expected_behavior
        if observed_behavior is None:
            observed_behavior = subject.observed_behavior
        reasons.extend(reproduction.reason_codes)
    elif minimized_evidence is not None or sealed_signals.minimized_evidence is not None:
        evidence_in = minimized_evidence
        if evidence_in is None:
            evidence_in = sealed_signals.minimized_evidence
        reproduction = _reproduction_from_prebuilt(
            evidence=evidence_in,
            reproduction_command=reproduction_command,
            reproduction_argv=tuple(reproduction_argv or ()),
        )
        reasons.extend(reproduction.reason_codes)
    else:
        raise DiagnosisRuntimeError(
            "diagnosis requires counterexample minimization inputs, a "
            "minimization_subject, or minimized_evidence",
            reason_code="missing_minimization_evidence",
        )

    if reproduction.minimization_failed:
        reasons.append(REASON_MINIMIZATION_FAILED)
    else:
        reasons.append(REASON_MINIMIZATION_SUCCEEDED)
    reasons.append(REASON_BOUNDED_REPRODUCTION)
    reasons.append(REASON_LOGS_BOUNDED)

    # --- phase: survivor report ---
    phases.append(DiagnosisPhase.SURVIVOR_REPORT.value)
    sealed_report: SurvivingMutantReport | None = None
    report_cid: str | None = None
    report_build: SurvivorReportBuildResult | None = None

    if survivor_report is not None:
        if isinstance(survivor_report, SurvivingMutantReport):
            sealed_report = survivor_report
        elif isinstance(survivor_report, Mapping):
            try:
                sealed_report = SurvivingMutantReport.from_dict(survivor_report)
            except (AnalysisContractError, TypeError, KeyError) as exc:
                raise DiagnosisRuntimeError(
                    f"survivor_report is malformed: {exc}"
                ) from exc
        else:
            raise DiagnosisRuntimeError(
                "survivor_report must be SurvivingMutantReport or mapping"
            )
        verify_survivor_report_identity(sealed_report)
        if not logs_remain_bounded(sealed_report):
            raise DiagnosisRuntimeError(
                "provided survivor_report must keep logs bounded",
                reason_code="logs_not_bounded",
            )
        report_cid = sealed_report.report_cid
        # Align reproduction evidence with report when caller supplied report.
        if sealed_report.minimized_evidence.minimization_failed != (
            reproduction.minimization_failed
        ):
            # Prefer the report's explicit failure flag for honesty.
            reproduction = BoundedReproduction(
                minimization_status=(
                    MinimizationStatus.FAILED.value
                    if sealed_report.minimized_evidence.minimization_failed
                    else MinimizationStatus.MINIMIZED.value
                ),
                minimization_failed=sealed_report.minimized_evidence.minimization_failed,
                minimization_failure_reason=(
                    sealed_report.minimized_evidence.notes
                    if sealed_report.minimized_evidence.minimization_failed
                    else None
                )
                or reproduction.minimization_failure_reason
                or (
                    "survivor report declared minimization failure"
                    if sealed_report.minimized_evidence.minimization_failed
                    else None
                ),
                evidence=sealed_report.minimized_evidence,
                reproduction_command=sealed_report.reproduction_command,
                reproduction_argv=reproduction.reproduction_argv,
                reproduction_input_cid=(
                    sealed_report.minimized_evidence.reproduction_input_cid
                ),
                guarantee=reproduction.guarantee,
                reason_codes=reproduction.reason_codes,
                bounded_log_digest=reproduction.bounded_log_digest
                or (
                    _bounded_digest_from_artifacts(
                        list(sealed_report.minimized_evidence.evidence_cids),
                        reason=sealed_report.minimized_evidence.notes
                        or "minimization failed",
                    )
                    if sealed_report.minimized_evidence.minimization_failed
                    else None
                ),
                quality=dict(reproduction.quality),
                logs_bounded=True,
                used_counterexample_minimizer=(
                    reproduction.used_counterexample_minimizer
                ),
            )
    else:
        exp = expected_behavior or (
            f"preserve {sealed_mutation.violated_or_missing_property}"
        )
        obs = observed_behavior or (
            f"survivor retained altered behavior for "
            f"{sealed_mutation.violated_or_missing_property}"
        )
        run_detectors = tuple(detectors_run) if detectors_run else ("unit.selected",)
        omitted = tuple(detectors_omitted)
        # Ensure detector inventory is disjoint.
        run_set = set(run_detectors)
        omitted = tuple(item for item in omitted if item not in run_set)
        report_build = _build_survivor_report(
            mutation=sealed_mutation,
            outcome=sealed_outcome,
            repository_state_cid=repo_state_cid,
            reproduction=reproduction,
            detectors_run=run_detectors,
            detectors_omitted=omitted,
            expected_behavior=exp,
            observed_behavior=obs,
            notes=notes,
            metadata=metadata,
        )
        sealed_report = SurvivingMutantReport.from_dict(dict(report_build.report))
        report_cid = sealed_report.report_cid

    # --- phase: semantic diagnosis ---
    phases.append(DiagnosisPhase.SEMANTIC_DIAGNOSIS.value)
    assert reproduction.evidence is not None
    signal_kwargs = {
        "equivalence_status": sealed_signals.equivalence_status,
        "intentionally_unconstrained": sealed_signals.intentionally_unconstrained,
        "specification_ambiguous": sealed_signals.specification_ambiguous,
        "product_defect_evidence": sealed_signals.product_defect_evidence,
        "original_behavior_violates_required_property": (
            sealed_signals.original_behavior_violates_required_property
        ),
        "product_defect_evidence_cids": sealed_signals.product_defect_evidence_cids,
        "difficulty_to_kill": sealed_signals.difficulty_to_kill,
        "observation_complete": True,
        "survivor_report_cid": report_cid,
        "minimized_evidence": reproduction.evidence,
        "not_selected_detector_ids": sealed_signals.not_selected_detector_ids,
        "not_executed_detector_ids": sealed_signals.not_executed_detector_ids,
        "path_unobserved_detector_ids": sealed_signals.path_unobserved_detector_ids,
        "weak_property_detector_ids": sealed_signals.weak_property_detector_ids,
        "dependency_omission_detector_ids": (
            sealed_signals.dependency_omission_detector_ids
        ),
        "capsule_omission_detector_ids": sealed_signals.capsule_omission_detector_ids,
        "comparison_result_cid": sealed_signals.comparison_result_cid,
        "primary_detector_kind": sealed_signals.primary_detector_kind,
        "notes": sealed_signals.notes,
        "metadata": {
            **dict(sealed_signals.metadata),
            "runtime_generator": GENERATOR_ID,
            "minimization_failed": reproduction.minimization_failed,
        },
    }
    diagnosis_signals = DiagnosisSignals(**signal_kwargs)  # type: ignore[arg-type]

    try:
        diagnosis = semantic_diagnose_surviving_mutant(
            sealed_mutation,
            sealed_outcome,
            repo_state_cid,
            signals=diagnosis_signals,
            comparison=sealed_comparison,
            notes=notes,
            metadata={
                **dict(metadata or {}),
                "orchestration": GENERATOR_ID,
                "datasets_interface": DATASETS_DIAGNOSE_INTERFACE,
                "diagnosis_step_order": list(DIAGNOSIS_STEP_ORDER),
            },
        )
    except SemanticDiagnosisError as exc:
        raise DiagnosisRuntimeError(
            f"semantic diagnosis failed: {exc}",
            reason_code="semantic_diagnosis_failed",
        ) from exc
    verify_survivor_diagnosis_identity(diagnosis)
    reasons.append(REASON_SEMANTIC_DIAGNOSIS)

    # --- phase: gap ---
    phases.append(DiagnosisPhase.GAP.value)
    gap: AssuranceGap | None = None
    gap_cid: str | None = None
    gap_status = GapPersistStatus.NOT_REQUIRED
    gap_persist_result: Mapping[str, Any] | None = None

    must_gap = high_risk or always_persist_gap
    if must_gap:
        reasons.append(REASON_HIGH_RISK_GAP_REQUIRED)
        gap = _build_assurance_gap(
            mutation=sealed_mutation,
            diagnosis=diagnosis,
            reproduction=reproduction,
            survivor_report_cid=report_cid,
            comparison=sealed_comparison,
            high_risk=high_risk,
        )
        gap_cid = gap.gap_cid
        reasons.append(REASON_GAP_IN_MEMORY)

        phases.append(DiagnosisPhase.PERSIST.value)
        persist_req: GapPersistRequest | None = None
        if gap_persist_request is not None:
            if isinstance(gap_persist_request, GapPersistRequest):
                persist_req = gap_persist_request
            elif isinstance(gap_persist_request, Mapping):
                persist_req = GapPersistRequest(
                    workspace=str(
                        gap_persist_request.get("workspace", "default")
                    ),
                    artifact_operation_id=str(
                        gap_persist_request.get(
                            "artifact_operation_id", "gap_artifact_op"
                        )
                    ),
                    history_operation_id=str(
                        gap_persist_request.get(
                            "history_operation_id", "gap_history_op"
                        )
                    ),
                    expected_history_generation=int(
                        gap_persist_request.get("expected_history_generation", 0)
                    ),
                    expected_history_head_cid=gap_persist_request.get(
                        "expected_history_head_cid"
                    ),
                    replicate=bool(gap_persist_request.get("replicate", False)),
                )
            else:
                raise DiagnosisRuntimeError(
                    "gap_persist_request must be GapPersistRequest or mapping"
                )
        gap_cid, gap_status, gap_persist_result = _persist_gap(
            gap,
            repository=gap_repository,
            request=persist_req,
        )
        if gap_status is GapPersistStatus.PERSISTED:
            reasons.append(REASON_GAP_PERSISTED)

    requires_review = bool(diagnosis.requires_human_review)
    human_accompanies = False
    if gap is not None:
        if gap.requires_human_review or gap.gap_class == (
            AssuranceGapClass.UNKNOWN.value
        ):
            requires_review = True
            human_accompanies = True
            reasons.append(REASON_HUMAN_REVIEW_ACCOMPANIES_GAP)
        if gap.gap_class == AssuranceGapClass.UNKNOWN.value:
            reasons.append(REASON_UNKNOWN_GAP_HUMAN_REVIEW)
    elif requires_review:
        # Non-high-risk unknown/ambiguous still records review without gap
        # replacement semantics (no gap present to replace).
        human_accompanies = False

    meta: dict[str, Any] = {
        **dict(metadata or {}),
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "adapter_id": ADAPTER_ID,
        "board_namespace": BOARD_NAMESPACE,
        "datasets_interface": DATASETS_DIAGNOSE_INTERFACE,
        "minimizer_interface": COUNTEREXAMPLE_MINIMIZER_INTERFACE,
        "high_risk_classes": list(
            high_risk_classes
            if high_risk_classes is not None
            else DEFAULT_HIGH_RISK_CLASSES
        ),
        "used_counterexample_minimizer": reproduction.used_counterexample_minimizer,
        "minimization_guarantee": reproduction.guarantee,
        "diagnosis_disposition": diagnosis.disposition,
        "deciding_step_id": diagnosis.deciding_step_id,
    }
    if counterexample_result is not None:
        meta["failure_identity_cid"] = counterexample_result.failure_identity_cid
        meta["counterexample_lease_ids"] = list(counterexample_result.lease_ids)
    if report_build is not None:
        meta["subject_observation_cid"] = report_build.subject_observation_cid

    diagnostic = diagnosis.summary
    if reproduction.minimization_failed:
        diagnostic = _clip(
            f"{diagnostic}; minimization_failed="
            f"{reproduction.minimization_failure_reason}",
            limit=MAX_DIAGNOSTIC,
        )

    return SurvivorDiagnosisRun(
        candidate_id=sealed_mutation.candidate_id,
        candidate_cid=sealed_mutation.candidate_cid,
        outcome_cid=sealed_outcome.outcome_cid,
        repository_state_cid=repo_state_cid,
        risk_class=risk,
        high_risk=high_risk,
        phases=tuple(phases),
        reproduction=reproduction,
        survivor_report=sealed_report,
        survivor_report_cid=report_cid,
        diagnosis=diagnosis,
        diagnosis_cid=diagnosis.diagnosis_cid,
        assurance_gap=gap,
        gap_cid=gap_cid,
        gap_persist_status=gap_status.value
        if isinstance(gap_status, GapPersistStatus)
        else str(gap_status),
        gap_persist_result=gap_persist_result,
        requires_human_review=requires_review,
        human_review_accompanies_gap=human_accompanies,
        minimization_failed=reproduction.minimization_failed,
        logs_bounded=True,
        production_policy_changed=False,
        reason_codes=_stable_unique(reasons),
        diagnostic=diagnostic,
        metadata=meta,
    )


def diagnose_surviving_mutant_descriptor() -> Mapping[str, Any]:
    """Return a static descriptor for the runtime diagnosis interface."""

    return MappingProxyType(
        {
            "interface_id": DIAGNOSE_SURVIVING_MUTANT_INTERFACE,
            "run_interface": SURVIVOR_DIAGNOSIS_RUN_INTERFACE,
            "evidence": AAE_DIAGNOSIS_RUN_EVIDENCE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "depends_on": (
                COUNTEREXAMPLE_MINIMIZER_INTERFACE,
                DATASETS_DIAGNOSE_INTERFACE,
                BUILD_SURVIVING_MUTANT_REPORT_INTERFACE,
                CLASSIFY_ASSURANCE_GAP_INTERFACE,
            ),
            "high_risk_classes": list(DEFAULT_HIGH_RISK_CLASSES),
            "diagnosis_step_order": list(DIAGNOSIS_STEP_ORDER),
            "production_policy_change": False,
        }
    )


__all__ = [
    "AAE_DIAGNOSIS_RUN_EVIDENCE",
    "ADAPTER_ID",
    "AssuranceGapRepository",
    "BOUNDED_REPRODUCTION_SCHEMA",
    "BoundedReproduction",
    "DEFAULT_HIGH_RISK_CLASSES",
    "DIAGNOSE_SURVIVING_MUTANT_INTERFACE",
    "DiagnosisPhase",
    "DiagnosisRuntimeError",
    "GapPersistRequest",
    "GapPersistStatus",
    "HighRiskGapMissingError",
    "InMemoryAssuranceGapRepository",
    "REASON_BOUNDED_REPRODUCTION",
    "REASON_COUNTEREXAMPLE_MINIMIZER",
    "REASON_GAP_IN_MEMORY",
    "REASON_GAP_PERSISTED",
    "REASON_HIGH_RISK_GAP_REQUIRED",
    "REASON_HUMAN_REVIEW_ACCOMPANIES_GAP",
    "REASON_LOGS_BOUNDED",
    "REASON_MINIMIZATION_FAILED",
    "REASON_MINIMIZATION_SKIPPED",
    "REASON_MINIMIZATION_SUCCEEDED",
    "REASON_NO_PRODUCTION_POLICY_CHANGE",
    "REASON_SEMANTIC_DIAGNOSIS",
    "REASON_UNKNOWN_GAP_HUMAN_REVIEW",
    "SURVIVOR_DIAGNOSIS_RUN_INTERFACE",
    "SURVIVOR_DIAGNOSIS_RUN_SCHEMA",
    "SurvivorDiagnosisRun",
    "diagnose_surviving_mutant",
    "diagnose_surviving_mutant_descriptor",
    "high_risk_classes",
    "is_high_risk",
    "run_counterexample_minimization",
]
