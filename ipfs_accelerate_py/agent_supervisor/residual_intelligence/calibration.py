"""Grouped calibration contracts for residual selective prediction.

Calibration is keyed by the exact family, repository, language, framework,
risk, model, quantization, hardware, and context-tier tuple.  There is no
global threshold.  Rows are held-out metrics and content identities only;
private bodies never enter the record.  Threshold mutation is an authorized
compare-and-swap with rollback, never a model self-modification.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    TrainingAvailability,
    UnknownFieldError,
    bounded_int,
    canonical_id,
    required_text,
    strict_fields,
    text_tuple,
)
from .rights import TrainingCorpusAdmission
from .splits import SplitPartition

CALIBRATION_GROUP_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-calibration-group@1"
CALIBRATION_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-calibration-evidence@1"
)
CALIBRATION_THRESHOLD_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-calibration-threshold-binding@1"
)
MAX_SCORE_PPM: Final = 1_000_000
MAX_CALIBRATION_EXAMPLES: Final = 20_000
MAX_CANDIDATE_THRESHOLDS: Final = 128
MAX_CALIBRATION_GROUPS: Final = 1_024
CALIBRATION_GROUP_AXES: Final[tuple[str, ...]] = (
    "family",
    "repository",
    "language",
    "framework",
    "risk",
    "model",
    "quantization",
    "hardware",
    "context_tier",
)
_FORBIDDEN_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "private_body",
        "raw_body",
        "hidden_test_body",
        "source_text",
        "prompt_text",
        "completion_text",
        "chain_of_thought",
    }
)


class ThresholdChangeOrigin(str, Enum):
    """Who requested a group threshold mutation.

    ``MODEL_SELF`` is recorded only so the compare-and-swap path can reject it.
    A binding can never persist that origin.
    """

    OPERATOR_CAS = "operator_cas"
    MODEL_SELF = "model_self"


def _axis_text(value: Any, name: str) -> str:
    return required_text(value, name, max_bytes=256)


def _ppm(value: Any, name: str) -> int:
    return bounded_int(value, name, minimum=0, maximum=MAX_SCORE_PPM)


def _count(value: Any, name: str, *, maximum: int = MAX_CALIBRATION_EXAMPLES) -> int:
    return bounded_int(value, name, minimum=0, maximum=maximum)


def _looks_like_private_body(value: str) -> bool:
    lowered = value.casefold()
    if lowered in _FORBIDDEN_BODY_MARKERS:
        return True
    head = lowered.split(":", 1)[0].split("/", 1)[0]
    if head in _FORBIDDEN_BODY_MARKERS:
        return True
    return any(
        lowered.startswith(marker + ":") or lowered.startswith(marker + "/")
        for marker in _FORBIDDEN_BODY_MARKERS
    )


def _reject_private_bodies(values: Sequence[str], *, noun: str) -> tuple[str, ...]:
    result = text_tuple(values, noun, max_items=MAX_CALIBRATION_EXAMPLES)
    if any(_looks_like_private_body(item) for item in result):
        raise ResidualIntelligenceError(
            f"{noun} exposes a private body rather than a content identity"
        )
    return result


def _threshold_candidates(values: Any) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError("evaluated_threshold_candidates must be a sequence")
    if len(values) > MAX_CANDIDATE_THRESHOLDS:
        raise ResidualIntelligenceError(
            f"evaluated_threshold_candidates exceeds {MAX_CANDIDATE_THRESHOLDS} items"
        )
    result = tuple(_ppm(item, "evaluated threshold candidate") for item in values)
    if len(set(result)) != len(result):
        raise ResidualIntelligenceError("evaluated_threshold_candidates contains duplicates")
    return result


@dataclass(frozen=True)
class CalibrationGroup:
    """Exact nine-axis calibration identity; never a global bucket."""

    family: ResidualTaskFamily
    repository: str
    language: str
    framework: str
    risk: RiskClass
    model: str
    quantization: str
    hardware: str
    context_tier: str
    schema: str = CALIBRATION_GROUP_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema", "group_id", "group_key", *CALIBRATION_GROUP_AXES}
    )

    def __post_init__(self) -> None:
        if self.schema != CALIBRATION_GROUP_SCHEMA:
            raise ResidualIntelligenceError("unsupported calibration group schema")
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        object.__setattr__(self, "risk", RiskClass(self.risk))
        for field in (
            "repository",
            "language",
            "framework",
            "model",
            "quantization",
            "hardware",
            "context_tier",
        ):
            object.__setattr__(self, field, _axis_text(getattr(self, field), field))

    @property
    def group_key(self) -> str:
        """Content identity of the nine grouping axes, excluding schema."""

        return canonical_id(self.axis_payload())

    @property
    def group_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def axis_payload(self) -> dict[str, str]:
        return {
            "family": self.family.value,
            "repository": self.repository,
            "language": self.language,
            "framework": self.framework,
            "risk": self.risk.value,
            "model": self.model,
            "quantization": self.quantization,
            "hardware": self.hardware,
            "context_tier": self.context_tier,
        }

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            **self.axis_payload(),
            "group_key": self.group_key,
        }
        if include_id:
            result["group_id"] = self.group_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibrationGroup:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"group_id", "group_key"},
            noun="calibration group",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            family=ResidualTaskFamily(str(payload.get("family") or "")),
            repository=str(payload.get("repository") or ""),
            language=str(payload.get("language") or ""),
            framework=str(payload.get("framework") or ""),
            risk=RiskClass(str(payload.get("risk") or "")),
            model=str(payload.get("model") or ""),
            quantization=str(payload.get("quantization") or ""),
            hardware=str(payload.get("hardware") or ""),
            context_tier=str(payload.get("context_tier") or ""),
        )
        claimed_key = str(payload.get("group_key") or "")
        if claimed_key and claimed_key != result.group_key:
            raise ResidualIntelligenceError("calibration group-key mismatch")
        claimed = str(payload.get("group_id") or "")
        if claimed and claimed != result.group_id:
            raise ResidualIntelligenceError("calibration group identity mismatch")
        return result


@dataclass(frozen=True)
class CalibrationEvidence:
    """Held-out metrics and CIDs for exactly one calibration group."""

    group: CalibrationGroup
    admission_id: str
    admission_decision: TrainingAvailability
    split_root: str
    holdout_root: str
    evaluation_identity: str
    example_identities: tuple[str, ...]
    adversarial_example_identities: tuple[str, ...]
    evaluated_threshold_candidates: tuple[int, ...]
    accept_count: int
    abstain_count: int
    reject_input_count: int
    ood_count: int
    capability_unavailable_count: int
    validation_required_count: int
    false_accept_count: int
    critical_false_accept_count: int
    precision_ppm: int
    abstention_rate_ppm: int
    partition: SplitPartition = SplitPartition.HELD_OUT
    hidden_test_bodies_accessed: bool = False
    schema: str = CALIBRATION_EVIDENCE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "evidence_id",
            "group",
            "admission_id",
            "admission_decision",
            "split_root",
            "holdout_root",
            "evaluation_identity",
            "example_identities",
            "adversarial_example_identities",
            "evaluated_threshold_candidates",
            "accept_count",
            "abstain_count",
            "reject_input_count",
            "ood_count",
            "capability_unavailable_count",
            "validation_required_count",
            "false_accept_count",
            "critical_false_accept_count",
            "precision_ppm",
            "abstention_rate_ppm",
            "partition",
            "hidden_test_bodies_accessed",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != CALIBRATION_EVIDENCE_SCHEMA:
            raise ResidualIntelligenceError("unsupported calibration evidence schema")
        if not isinstance(self.group, CalibrationGroup):
            raise ResidualIntelligenceError("calibration evidence requires a typed group")
        object.__setattr__(self, "admission_decision", TrainingAvailability(self.admission_decision))
        object.__setattr__(self, "partition", SplitPartition(self.partition))
        for field in ("admission_id", "split_root", "holdout_root", "evaluation_identity"):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        identities = _reject_private_bodies(self.example_identities, noun="example_identities")
        if not identities:
            raise ResidualIntelligenceError(
                "calibration evidence requires held-out example identities"
            )
        object.__setattr__(self, "example_identities", identities)
        object.__setattr__(
            self,
            "adversarial_example_identities",
            _reject_private_bodies(
                self.adversarial_example_identities,
                noun="adversarial_example_identities",
            ),
        )
        object.__setattr__(
            self,
            "evaluated_threshold_candidates",
            _threshold_candidates(self.evaluated_threshold_candidates),
        )
        if self.partition is not SplitPartition.HELD_OUT:
            raise ResidualIntelligenceError("calibration rows must be held-out")
        if self.admission_decision is not TrainingAvailability.ADMITTED:
            raise ResidualIntelligenceError(
                "calibration rows require an admitted TrainingCorpusAdmission"
            )
        if type(self.hidden_test_bodies_accessed) is not bool:
            raise ResidualIntelligenceError("hidden_test_bodies_accessed must be boolean")
        if self.hidden_test_bodies_accessed:
            raise ResidualIntelligenceError("calibration evidence cannot access hidden-test bodies")
        n_examples = len(identities)
        counts = {}
        for field in (
            "accept_count",
            "abstain_count",
            "reject_input_count",
            "ood_count",
            "capability_unavailable_count",
            "validation_required_count",
            "false_accept_count",
            "critical_false_accept_count",
        ):
            counts[field] = _count(getattr(self, field), field, maximum=n_examples)
            object.__setattr__(self, field, counts[field])
        disposition_total = (
            counts["accept_count"]
            + counts["abstain_count"]
            + counts["reject_input_count"]
            + counts["ood_count"]
            + counts["capability_unavailable_count"]
            + counts["validation_required_count"]
        )
        if disposition_total != n_examples:
            raise ResidualIntelligenceError(
                "disposition counts must equal the held-out example population"
            )
        if counts["false_accept_count"] > counts["accept_count"]:
            raise ResidualIntelligenceError("false accepts cannot exceed accepts")
        if counts["critical_false_accept_count"] > counts["false_accept_count"]:
            raise ResidualIntelligenceError("critical false accepts cannot exceed false accepts")
        true_accepts = counts["accept_count"] - counts["false_accept_count"]
        derived_precision = (
            0
            if counts["accept_count"] == 0
            else (true_accepts * MAX_SCORE_PPM) // counts["accept_count"]
        )
        derived_abstention = (counts["abstain_count"] * MAX_SCORE_PPM) // n_examples
        object.__setattr__(self, "precision_ppm", _ppm(self.precision_ppm, "precision_ppm"))
        object.__setattr__(
            self,
            "abstention_rate_ppm",
            _ppm(self.abstention_rate_ppm, "abstention_rate_ppm"),
        )
        if self.precision_ppm != derived_precision:
            raise ResidualIntelligenceError("precision_ppm does not match held-out counts")
        if self.abstention_rate_ppm != derived_abstention:
            raise ResidualIntelligenceError("abstention_rate_ppm does not match held-out counts")

    @property
    def evidence_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def n_examples(self) -> int:
        return len(self.example_identities)

    @property
    def includes_adversarial_evaluation(self) -> bool:
        return bool(self.adversarial_example_identities)

    @property
    def zero_critical_false_accepts(self) -> bool:
        return self.critical_false_accept_count == 0

    def is_current(
        self,
        *,
        admission_id: str,
        split_root: str,
        evaluation_identity: str,
        holdout_root: str = "",
    ) -> bool:
        if (
            self.admission_id != admission_id
            or self.split_root != split_root
            or self.evaluation_identity != evaluation_identity
        ):
            return False
        if holdout_root and self.holdout_root != holdout_root:
            return False
        return (
            self.admission_decision is TrainingAvailability.ADMITTED
            and self.partition is SplitPartition.HELD_OUT
            and not self.hidden_test_bodies_accessed
        )

    def eligible_for_threshold_cas(
        self,
        *,
        admission_id: str,
        split_root: str,
        evaluation_identity: str,
        holdout_root: str = "",
        proposed_threshold_ppm: int,
    ) -> None:
        if not self.is_current(
            admission_id=admission_id,
            split_root=split_root,
            evaluation_identity=evaluation_identity,
            holdout_root=holdout_root,
        ):
            raise ResidualIntelligenceError("threshold change requires current evidence")
        if not self.includes_adversarial_evaluation:
            raise ResidualIntelligenceError(
                "threshold change requires current held-out and adversarial evaluation"
            )
        if not self.zero_critical_false_accepts:
            raise ResidualIntelligenceError(
                "threshold change requires zero critical false accepts"
            )
        if proposed_threshold_ppm not in self.evaluated_threshold_candidates:
            raise ResidualIntelligenceError(
                "proposed threshold was not evaluated on current group evidence"
            )

    def validate_against_admission(self, admission: TrainingCorpusAdmission) -> None:
        if not isinstance(admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("admission must be TrainingCorpusAdmission")
        if not admission.can_train:
            raise ResidualIntelligenceError(
                "calibration rows require an admitted TrainingCorpusAdmission"
            )
        if self.admission_id != admission.admission_id:
            raise ResidualIntelligenceError("calibration evidence admission_id mismatch")
        if self.split_root != admission.split_root:
            raise ResidualIntelligenceError("calibration evidence split_root mismatch")
        if self.holdout_root not in admission.holdout_roots:
            raise ResidualIntelligenceError(
                "calibration holdout_root is not covered by the admitted holdout roots"
            )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "group": self.group.to_dict(),
            "admission_id": self.admission_id,
            "admission_decision": self.admission_decision.value,
            "split_root": self.split_root,
            "holdout_root": self.holdout_root,
            "evaluation_identity": self.evaluation_identity,
            "example_identities": list(self.example_identities),
            "adversarial_example_identities": list(self.adversarial_example_identities),
            "evaluated_threshold_candidates": list(self.evaluated_threshold_candidates),
            "accept_count": self.accept_count,
            "abstain_count": self.abstain_count,
            "reject_input_count": self.reject_input_count,
            "ood_count": self.ood_count,
            "capability_unavailable_count": self.capability_unavailable_count,
            "validation_required_count": self.validation_required_count,
            "false_accept_count": self.false_accept_count,
            "critical_false_accept_count": self.critical_false_accept_count,
            "precision_ppm": self.precision_ppm,
            "abstention_rate_ppm": self.abstention_rate_ppm,
            "partition": self.partition.value,
            "hidden_test_bodies_accessed": False,
        }
        if include_id:
            result["evidence_id"] = self.evidence_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibrationEvidence:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"evidence_id"},
            noun="calibration evidence",
        )
        group_payload = payload.get("group")
        if not isinstance(group_payload, Mapping):
            raise ResidualIntelligenceError("calibration evidence group must be an object")
        result = cls(
            schema=str(payload.get("schema") or ""),
            group=CalibrationGroup.from_dict(group_payload),
            admission_id=str(payload.get("admission_id") or ""),
            admission_decision=TrainingAvailability(str(payload.get("admission_decision") or "")),
            split_root=str(payload.get("split_root") or ""),
            holdout_root=str(payload.get("holdout_root") or ""),
            evaluation_identity=str(payload.get("evaluation_identity") or ""),
            example_identities=tuple(payload.get("example_identities") or ()),
            adversarial_example_identities=tuple(
                payload.get("adversarial_example_identities") or ()
            ),
            evaluated_threshold_candidates=tuple(
                payload.get("evaluated_threshold_candidates") or ()
            ),
            accept_count=payload.get("accept_count"),
            abstain_count=payload.get("abstain_count"),
            reject_input_count=payload.get("reject_input_count"),
            ood_count=payload.get("ood_count"),
            capability_unavailable_count=payload.get("capability_unavailable_count"),
            validation_required_count=payload.get("validation_required_count"),
            false_accept_count=payload.get("false_accept_count"),
            critical_false_accept_count=payload.get("critical_false_accept_count"),
            precision_ppm=payload.get("precision_ppm"),
            abstention_rate_ppm=payload.get("abstention_rate_ppm"),
            partition=SplitPartition(str(payload.get("partition") or "")),
            hidden_test_bodies_accessed=payload.get("hidden_test_bodies_accessed"),
        )
        claimed = str(payload.get("evidence_id") or "")
        if claimed and claimed != result.evidence_id:
            raise ResidualIntelligenceError("calibration evidence identity mismatch")
        return result


@dataclass(frozen=True)
class CalibrationThresholdBinding:
    """One group's accept threshold, mutated only by authorized CAS."""

    group_key: str
    accept_threshold_ppm: int
    evidence_id: str
    cas_identity: str
    origin: ThresholdChangeOrigin
    previous_binding_id: str = ""
    rollback_threshold_ppm: int = 0
    schema: str = CALIBRATION_THRESHOLD_BINDING_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "binding_id",
            "group_key",
            "accept_threshold_ppm",
            "evidence_id",
            "cas_identity",
            "origin",
            "previous_binding_id",
            "rollback_threshold_ppm",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != CALIBRATION_THRESHOLD_BINDING_SCHEMA:
            raise ResidualIntelligenceError("unsupported calibration threshold binding schema")
        object.__setattr__(self, "group_key", required_text(self.group_key, "group_key"))
        object.__setattr__(
            self,
            "accept_threshold_ppm",
            _ppm(self.accept_threshold_ppm, "accept_threshold_ppm"),
        )
        object.__setattr__(self, "evidence_id", required_text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "cas_identity", required_text(self.cas_identity, "cas_identity"))
        object.__setattr__(self, "origin", ThresholdChangeOrigin(self.origin))
        if self.origin is ThresholdChangeOrigin.MODEL_SELF:
            raise ResidualIntelligenceError(
                "self-threshold-rejection: a model cannot change its threshold"
            )
        object.__setattr__(
            self,
            "previous_binding_id",
            ""
            if self.previous_binding_id in (None, "")
            else required_text(self.previous_binding_id, "previous_binding_id"),
        )
        object.__setattr__(
            self,
            "rollback_threshold_ppm",
            _ppm(self.rollback_threshold_ppm, "rollback_threshold_ppm"),
        )

    @property
    def binding_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "group_key": self.group_key,
            "accept_threshold_ppm": self.accept_threshold_ppm,
            "evidence_id": self.evidence_id,
            "cas_identity": self.cas_identity,
            "origin": self.origin.value,
            "previous_binding_id": self.previous_binding_id,
            "rollback_threshold_ppm": self.rollback_threshold_ppm,
        }
        if include_id:
            result["binding_id"] = self.binding_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibrationThresholdBinding:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"binding_id", "previous_binding_id", "rollback_threshold_ppm"},
            noun="calibration threshold binding",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            group_key=str(payload.get("group_key") or ""),
            accept_threshold_ppm=payload.get("accept_threshold_ppm"),
            evidence_id=str(payload.get("evidence_id") or ""),
            cas_identity=str(payload.get("cas_identity") or ""),
            origin=ThresholdChangeOrigin(str(payload.get("origin") or "")),
            previous_binding_id=str(payload.get("previous_binding_id") or ""),
            rollback_threshold_ppm=payload.get("rollback_threshold_ppm") or 0,
        )
        claimed = str(payload.get("binding_id") or "")
        if claimed and claimed != result.binding_id:
            raise ResidualIntelligenceError("calibration threshold binding identity mismatch")
        return result


def apply_threshold_cas(
    current: CalibrationThresholdBinding | None,
    *,
    group: CalibrationGroup,
    evidence: CalibrationEvidence,
    proposed_threshold_ppm: int,
    origin: ThresholdChangeOrigin | str,
    cas_identity: str,
    expected_binding_id: str = "",
    admission_id: str,
    split_root: str,
    evaluation_identity: str,
    holdout_root: str = "",
) -> CalibrationThresholdBinding:
    """Install a group threshold through authorized compare-and-swap."""

    if not isinstance(group, CalibrationGroup):
        raise ResidualIntelligenceError("threshold CAS requires a typed calibration group")
    if not isinstance(evidence, CalibrationEvidence):
        raise ResidualIntelligenceError("threshold CAS requires typed calibration evidence")
    if evidence.group.group_key != group.group_key:
        raise ResidualIntelligenceError("threshold CAS evidence is not isolated to the group-key")
    origin_value = ThresholdChangeOrigin(origin)
    if origin_value is ThresholdChangeOrigin.MODEL_SELF:
        raise ResidualIntelligenceError(
            "self-threshold-rejection: a model cannot change its threshold"
        )
    current_id = current.binding_id if current is not None else ""
    if expected_binding_id in (None, ""):
        expected = ""
    else:
        expected = required_text(expected_binding_id, "expected_binding_id")
    if expected != current_id:
        raise ResidualIntelligenceError("stale calibration threshold compare-and-swap")
    if current is not None and current.group_key != group.group_key:
        raise ResidualIntelligenceError("threshold CAS current binding group-key mismatch")
    evidence.eligible_for_threshold_cas(
        admission_id=admission_id,
        split_root=split_root,
        evaluation_identity=evaluation_identity,
        holdout_root=holdout_root,
        proposed_threshold_ppm=proposed_threshold_ppm,
    )
    rollback = current.accept_threshold_ppm if current is not None else proposed_threshold_ppm
    return CalibrationThresholdBinding(
        group_key=group.group_key,
        accept_threshold_ppm=proposed_threshold_ppm,
        evidence_id=evidence.evidence_id,
        cas_identity=cas_identity,
        origin=ThresholdChangeOrigin.OPERATOR_CAS,
        previous_binding_id=current_id,
        rollback_threshold_ppm=rollback,
    )


def rollback_threshold_binding(
    binding: CalibrationThresholdBinding,
    *,
    cas_identity: str,
) -> CalibrationThresholdBinding:
    """Restore the prior admitted group threshold by authorized CAS."""

    if not isinstance(binding, CalibrationThresholdBinding):
        raise ResidualIntelligenceError("rollback requires a typed threshold binding")
    if not binding.previous_binding_id:
        raise ResidualIntelligenceError("genesis threshold binding has no rollback target")
    return CalibrationThresholdBinding(
        group_key=binding.group_key,
        accept_threshold_ppm=binding.rollback_threshold_ppm,
        evidence_id=binding.evidence_id,
        cas_identity=cas_identity,
        origin=ThresholdChangeOrigin.OPERATOR_CAS,
        previous_binding_id=binding.binding_id,
        rollback_threshold_ppm=binding.accept_threshold_ppm,
    )


def reject_global_threshold_fields(payload: Mapping[str, Any], *, noun: str) -> None:
    """Fail closed if a record tries to carry a cross-group threshold."""

    if not isinstance(payload, Mapping):
        raise ResidualIntelligenceError(f"{noun} must be an object")
    forbidden = []
    for key in payload:
        normalized = str(key).strip().casefold().replace("-", "_")
        if normalized in {
            "global_threshold",
            "global_threshold_ppm",
            "default_threshold",
            "default_threshold_ppm",
            "shared_threshold",
            "shared_threshold_ppm",
        }:
            forbidden.append(str(key))
    if forbidden:
        raise UnknownFieldError(
            f"{noun} contains a global threshold field: {', '.join(sorted(forbidden))}"
        )


__all__ = (
    "CALIBRATION_EVIDENCE_SCHEMA",
    "CALIBRATION_GROUP_AXES",
    "CALIBRATION_GROUP_SCHEMA",
    "CALIBRATION_THRESHOLD_BINDING_SCHEMA",
    "MAX_CALIBRATION_EXAMPLES",
    "MAX_CALIBRATION_GROUPS",
    "MAX_CANDIDATE_THRESHOLDS",
    "MAX_SCORE_PPM",
    "CalibrationEvidence",
    "CalibrationGroup",
    "CalibrationThresholdBinding",
    "ThresholdChangeOrigin",
    "apply_threshold_cas",
    "reject_global_threshold_fields",
    "rollback_threshold_binding",
)
