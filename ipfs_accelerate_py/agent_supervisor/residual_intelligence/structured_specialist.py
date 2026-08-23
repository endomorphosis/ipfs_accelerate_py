"""Family-bounded grammar-constrained structured specialist.

One expert instance covers exactly one of the two admitted structured
families: typed procedure-hole resolution or patch-sketch generation.
The specialist is a provider-free constrained decoder adapter.  It emits
and consumes only the VRIF-008 family grammar, strictly post-parses every
candidate against bounded compact context, treats every parse or bound
failure as ``invalid_output`` with no prose recovery, and never creates
authority, policy, completion, or freeform fields.  Fine-tuning is
forbidden until an admitted ``TrainingCorpusAdmission`` is bound; missing
training does not invent weights or download a model.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from .abstention import (
    AbstentionDecision,
    SelectivePredictionPolicy,
    SelectivePredictionRequest,
    selectively_predict,
)
from .calibration import CalibrationGroup, MAX_SCORE_PPM
from .contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    TrainingAvailability,
    bounded_int,
    bounded_json_mapping,
    canonical_id,
    reject_candidate_authority,
    reject_secret_material,
    required_text,
    strict_fields,
    text_tuple,
)
from .expert_specs import (
    EXPERT_CLASS_FORMS,
    ExpertClass,
    ResidualExpertSpec,
    expert_spec_for,
    parse_expert_class,
)
from .local_experts import IndependentValidationReceipt
from .residual_ir import ResidualIntelligenceIR, ResidualTaskInput, ResidualTaskOutput
from .rights import TrainingCorpusAdmission
from .structured_decoding import (
    DecodeStatus,
    ExpertGrammar,
    StructuredDecodeResult,
    decode_structured_output,
    grammar_for,
)
from .task_families import (
    ABSTAIN_OUTPUT_CLASS,
    CANDIDATE_ONLY_AUTHORITY,
    ERROR_INVALID_OUTPUT,
    REASON_MISSING_COMPACT_FEATURE,
    REASON_TASK_FAMILY_MISMATCH,
    REASON_TOKEN_LIMIT,
    REASON_UNKNOWN_COMPACT_FEATURE,
    REASON_UNSUPPORTED_FAMILY_RISK,
    ResidualTaskFamilySpec,
    family_spec_for,
)

STRUCTURED_DECODE_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-structured-decode-request@1"
)
STRUCTURED_SPECIALIST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-constrained-structured-expert@1"
)
STRUCTURED_SPECIALIST_PREDICTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-structured-specialist-prediction@1"
)
CONSTRAINED_DECODER_FORM: Final = "constrained_structured_decoder"
ADMITTED_STRUCTURED_FAMILIES: Final[frozenset[ResidualTaskFamily]] = frozenset(
    {
        ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
        ResidualTaskFamily.PATCH_SKETCH_GENERATION,
    }
)
DEFAULT_HOLE_OPERATORS: Final[tuple[str, ...]] = (
    "bind_argument",
    "select_procedure",
    "instantiate_hole",
    "apply_rewrite",
    "fill_literal",
)
DEFAULT_PATCH_OPERATIONS: Final[tuple[str, ...]] = (
    "replace_function",
    "replace_method",
    "insert_statement",
    "update_binding",
    "add_guard",
    "narrow_type",
    "repair_call",
)
FORBIDDEN_FREEFORM_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "explanation",
        "rationale",
        "commentary",
        "prose",
        "shell",
        "bash",
        "sudo",
        "curl",
        "wget",
        "rm",
        "chmod",
        "policy",
        "authority",
        "authorization",
        "completion",
        "completed",
        "promote",
        "promotion",
    }
)
FORBIDDEN_PATCH_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "delete_test",
        "weaken_validation",
        "change_authority",
        "chmod",
        "shell",
        "rm",
        "sudo",
        "curl",
        "wget",
        "bash",
        "arbitrary_shell",
    }
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
        "private_chain_of_thought",
    }
)
PROPOSAL_RISKS: Final[frozenset[RiskClass]] = frozenset({RiskClass.R4, RiskClass.R5})
MAX_SPECIALIST_EXAMPLES: Final = 30_000
MAX_SPECIALIST_STEPS: Final = 8_000
MAX_SPECIALIST_WALL_SECONDS: Final = 10_800
MAX_SPECIALIST_GPU_SECONDS: Final = 7_200
MAX_SPECIALIST_CHECKPOINTS: Final = 3
MAX_VOCABULARY_ITEMS: Final = 64
DEFAULT_PATCH_LINE_BOUND: Final = 200
DEFAULT_SCORE_PPM: Final = 900_000

REASON_INVALID_OUTPUT: Final = ERROR_INVALID_OUTPUT
REASON_MAX_LENGTH: Final = "max_length_exceeded"
REASON_FAMILY_MISMATCH: Final = REASON_TASK_FAMILY_MISMATCH
REASON_UNSUPPORTED_FAMILY: Final = "unsupported_structured_family"
REASON_UNSUPPORTED_CLASS: Final = "structured_specialist_requires_class_e"
REASON_REJECT_INPUT: Final = "reject_input"
REASON_COMPILER_UNAVAILABLE: Final = "compiler_capability_unavailable"
REASON_DECODER_UNAVAILABLE: Final = "constrained_decoder_unavailable"
REASON_PRECONDITIONS: Final = "procedure_preconditions_unsatisfied"
REASON_INCOMPLETE_CONTEXT: Final = "bounded_context_insufficient"
REASON_OUT_OF_SCOPE: Final = "out_of_scope_path"
REASON_TEST_DELETION: Final = "test_deletion_forbidden"
REASON_VALIDATION_WEAKENING: Final = "validation_weakening_forbidden"
REASON_FREEFORM_AUTHORITY: Final = "freeform_authority_forbidden"
REASON_ABSTAIN_ESCALATE: Final = "abstain_escalate"
REASON_TRAINING_UNAVAILABLE: Final = "training_unavailable"
REASON_INDEPENDENT_VALIDATOR: Final = "independent_validator_required"
REASON_VALIDATION_REQUIRED: Final = "validation_required"
REASON_IR_VALIDATION_REQUIRED: Final = ExpertDisposition.VALIDATION_REQUIRED.value
REASON_R4_R5_PROPOSAL: Final = "r4_r5_proposal_tier"
REASON_NO_GROUP_THRESHOLD: Final = "no_group_threshold"
REASON_CURRENT_EVIDENCE: Final = "current_evidence_required"
REASON_GRAMMAR_MISMATCH: Final = "grammar_identity_mismatch"


class StructuredSpecialistForm(str, Enum):
    """Closed producer used for one structured-specialist prediction."""

    CONSTRAINED_STRUCTURED_DECODER = CONSTRAINED_DECODER_FORM
    ABSTAIN = "abstain"
    REJECT_INPUT = "reject_input"


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _ppm(value: Any, name: str) -> int:
    return bounded_int(value, name, minimum=0, maximum=MAX_SCORE_PPM)


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


def _reject_private_name(value: str, name: str) -> str:
    text = required_text(value, name, max_bytes=256)
    if _looks_like_private_body(text):
        raise ResidualIntelligenceError(f"{name} cannot memorize or expose a private body")
    return text


def _closed_vocabulary(values: Any, name: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if values in (None, ()):
        items = default
    else:
        items = text_tuple(values, name, allow_empty=False, max_items=MAX_VOCABULARY_ITEMS)
    if any(_looks_like_private_body(item) for item in items):
        raise ResidualIntelligenceError(f"{name} cannot memorize or expose a private body")
    forbidden = sorted(
        item
        for item in items
        if item.casefold() in FORBIDDEN_FREEFORM_MARKERS
        or item.casefold() in FORBIDDEN_PATCH_OPERATIONS
    )
    if forbidden:
        raise ResidualIntelligenceError(
            f"{name} contains freeform or authority operations: {', '.join(forbidden)}"
        )
    return items


def _utf8_len(raw: str | bytes) -> int:
    if isinstance(raw, bytes):
        return len(raw)
    return len(raw.encode("utf-8"))


def _token_estimate(raw: str | bytes) -> int:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    parts = text.split()
    return max(len(parts), 1 if text else 0)


def _as_token_list(value: Any, name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if isinstance(value, str):
        item = _reject_private_name(value, name)
        return (item,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = tuple(_reject_private_name(str(item), f"{name} item") for item in value)
        if not allow_empty and not items:
            raise ResidualIntelligenceError(f"{name} must not be empty")
        if len(set(items)) != len(items):
            raise ResidualIntelligenceError(f"{name} contains duplicate values")
        return items
    raise ResidualIntelligenceError(f"{name} must be a token or token list")


def _contains_forbidden_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in FORBIDDEN_FREEFORM_MARKERS or normalized in FORBIDDEN_PATCH_OPERATIONS:
                return True
            if _contains_forbidden_marker(child):
                return True
        return False
    if isinstance(value, str):
        lowered = value.strip().casefold().replace("-", "_")
        if lowered in FORBIDDEN_FREEFORM_MARKERS or lowered in FORBIDDEN_PATCH_OPERATIONS:
            return True
        return any(marker in lowered for marker in ("rm_-rf", "sudo", "/bin/sh"))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return any(_contains_forbidden_marker(item) for item in value)
    return False


def _invalid_decode(grammar: ExpertGrammar) -> StructuredDecodeResult:
    return StructuredDecodeResult(
        status=DecodeStatus.INVALID_OUTPUT,
        grammar_id=grammar.grammar_id,
        output=None,
        reason_codes=(REASON_INVALID_OUTPUT,),
    )


def _unique_reasons(*groups: Sequence[str]) -> tuple[str, ...]:
    stamped: list[str] = []
    for group in groups:
        for item in group:
            if item and item not in stamped:
                stamped.append(item)
    return tuple(stamped)


def _validation_required_reasons(
    reasons: Sequence[str],
    *,
    risk: RiskClass,
    extra: Sequence[str] = (),
) -> tuple[str, ...]:
    """Stamp both the policy token and the IR-required R4/R5 token."""

    stamped = _unique_reasons(reasons, extra)
    required = [REASON_VALIDATION_REQUIRED, REASON_IR_VALIDATION_REQUIRED]
    if risk in PROPOSAL_RISKS:
        required.append(REASON_R4_R5_PROPOSAL)
    return _unique_reasons(stamped, required)


def _keep_proposal_candidate(
    candidate: ResidualTaskOutput,
    *,
    reasons: Sequence[str],
) -> ResidualTaskOutput:
    return ResidualTaskOutput(
        output_class=candidate.output_class,
        structured_payload=candidate.structured_payload,
        confidence_or_score=candidate.confidence_or_score,
        calibration_group=candidate.calibration_group,
        abstained=False,
        reason_codes=tuple(reasons),
        evidence_references=candidate.evidence_references,
        candidate_only=True,
    )


def _envelope(
    *,
    output_class: str,
    payload: Mapping[str, Any],
    score_ppm: int,
    calibration_group: str,
    abstained: bool,
    reason_codes: Sequence[str],
    evidence_references: Sequence[str],
) -> str:
    return json.dumps(
        {
            "output_class": output_class,
            "structured_payload": dict(payload),
            "confidence_or_score": score_ppm,
            "calibration_group": calibration_group,
            "abstained": abstained,
            "reason_codes": list(reason_codes),
            "evidence_references": list(evidence_references),
            "candidate_only": True,
        },
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=False,
    )


@dataclass(frozen=True)
class StructuredDecodeRequest:
    """Bounded decode request: compact task input plus optional decoder bytes."""

    task_input: ResidualTaskInput
    raw_output: str = ""
    independent_validation: IndependentValidationReceipt | None = None
    grammar_id: str = ""
    candidate_only: bool = True
    schema: str = STRUCTURED_DECODE_REQUEST_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "request_id",
            "task_input",
            "raw_output",
            "independent_validation",
            "grammar_id",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != STRUCTURED_DECODE_REQUEST_SCHEMA:
            raise ResidualIntelligenceError("unsupported structured decode request schema")
        if not isinstance(self.task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("structured decode request requires ResidualTaskInput")
        raw = self.raw_output
        if raw is None:
            object.__setattr__(self, "raw_output", "")
        elif isinstance(raw, bytes):
            try:
                object.__setattr__(self, "raw_output", raw.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ResidualIntelligenceError("decoder output must be UTF-8") from exc
        elif isinstance(raw, str):
            object.__setattr__(self, "raw_output", raw)
        else:
            raise ResidualIntelligenceError("raw_output must be UTF-8 text or bytes")
        if self.independent_validation is not None and not isinstance(
            self.independent_validation, IndependentValidationReceipt
        ):
            raise ResidualIntelligenceError(
                "independent_validation must be IndependentValidationReceipt"
            )
        object.__setattr__(
            self,
            "grammar_id",
            ""
            if self.grammar_id in (None, "")
            else required_text(self.grammar_id, "grammar_id"),
        )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("structured decode requests must remain candidate_only=true")
        reject_secret_material(self.task_input.compact_features, noun="compact_features")
        for key in self.task_input.compact_features:
            _reject_private_name(str(key), "compact_features key")

    @property
    def request_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def has_decoder_emission(self) -> bool:
        return self.raw_output != ""

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_input": self.task_input.to_dict(),
            "raw_output": self.raw_output,
            "independent_validation": (
                None
                if self.independent_validation is None
                else self.independent_validation.to_dict()
            ),
            "grammar_id": self.grammar_id,
            "candidate_only": True,
        }
        if include_id:
            result["request_id"] = self.request_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StructuredDecodeRequest:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {"request_id", "raw_output", "independent_validation", "grammar_id"},
            noun="structured decode request",
        )
        validation_payload = payload.get("independent_validation")
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_input=ResidualTaskInput.from_dict(payload.get("task_input") or {}),
            raw_output=str(payload.get("raw_output") or ""),
            independent_validation=(
                None
                if validation_payload in (None, {})
                else IndependentValidationReceipt.from_dict(validation_payload)
            ),
            grammar_id=str(payload.get("grammar_id") or ""),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("request_id") or "")
        if claimed and claimed != result.request_id:
            raise ResidualIntelligenceError("structured decode request identity mismatch")
        return result


@dataclass(frozen=True)
class StructuredSpecialistPrediction:
    """Candidate-only specialist result with strict decode status and disposition."""

    decode_result: StructuredDecodeResult
    task_output: ResidualTaskOutput
    form: StructuredSpecialistForm
    disposition: ExpertDisposition
    feature_identity: str
    abstention: AbstentionDecision | None = None
    independent_validator_identity: str = ""
    structured_valid: bool = False
    model_calls: int = 0
    provider_invocations: int = 0
    candidate_only: bool = True
    schema: str = STRUCTURED_SPECIALIST_PREDICTION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "prediction_id",
            "decode_result",
            "task_output",
            "form",
            "disposition",
            "feature_identity",
            "abstention",
            "independent_validator_identity",
            "structured_valid",
            "model_calls",
            "provider_invocations",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != STRUCTURED_SPECIALIST_PREDICTION_SCHEMA:
            raise ResidualIntelligenceError("unsupported structured specialist prediction schema")
        if not isinstance(self.decode_result, StructuredDecodeResult):
            raise ResidualIntelligenceError("decode_result must be StructuredDecodeResult")
        if not isinstance(self.task_output, ResidualTaskOutput):
            raise ResidualIntelligenceError("task_output must be ResidualTaskOutput")
        object.__setattr__(self, "form", StructuredSpecialistForm(self.form))
        object.__setattr__(self, "disposition", ExpertDisposition(self.disposition))
        object.__setattr__(
            self, "feature_identity", required_text(self.feature_identity, "feature_identity")
        )
        if self.abstention is not None and not isinstance(self.abstention, AbstentionDecision):
            raise ResidualIntelligenceError("abstention must be AbstentionDecision")
        object.__setattr__(
            self, "structured_valid", _require_bool(self.structured_valid, "structured_valid")
        )
        object.__setattr__(
            self,
            "independent_validator_identity",
            ""
            if self.independent_validator_identity in (None, "")
            else required_text(
                self.independent_validator_identity, "independent_validator_identity"
            ),
        )
        object.__setattr__(self, "model_calls", bounded_int(self.model_calls, "model_calls", minimum=0, maximum=0))
        object.__setattr__(
            self,
            "provider_invocations",
            bounded_int(self.provider_invocations, "provider_invocations", minimum=0, maximum=0),
        )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if self.task_output.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if self.decode_result.status is DecodeStatus.INVALID_OUTPUT and self.decode_result.output is not None:
            raise ResidualIntelligenceError("invalid_output cannot carry a parsed candidate")
        if self.disposition is ExpertDisposition.ACCEPT:
            if self.task_output.abstained:
                raise ResidualIntelligenceError("ACCEPT cannot be abstained")
            if not self.structured_valid:
                raise ResidualIntelligenceError("ACCEPT requires structured validity")
            if not self.independent_validator_identity:
                raise ResidualIntelligenceError("ACCEPT requires an independent validator")
            if self.decode_result.status is not DecodeStatus.VALID:
                raise ResidualIntelligenceError("ACCEPT requires a valid constrained decode")

    @property
    def prediction_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def as_ir(self, task_input: ResidualTaskInput) -> ResidualIntelligenceIR:
        return ResidualIntelligenceIR(task_input, self.task_output)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "decode_result": self.decode_result.to_dict(),
            "task_output": self.task_output.to_dict(),
            "form": self.form.value,
            "disposition": self.disposition.value,
            "feature_identity": self.feature_identity,
            "abstention": None if self.abstention is None else self.abstention.to_dict(),
            "independent_validator_identity": self.independent_validator_identity,
            "structured_valid": self.structured_valid,
            "model_calls": 0,
            "provider_invocations": 0,
            "candidate_only": True,
        }
        if include_id:
            result["prediction_id"] = self.prediction_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StructuredSpecialistPrediction:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {"prediction_id", "abstention", "independent_validator_identity"},
            noun="structured specialist prediction",
        )
        abstention_payload = payload.get("abstention")
        result = cls(
            schema=str(payload.get("schema") or ""),
            decode_result=StructuredDecodeResult(
                status=DecodeStatus(str((payload.get("decode_result") or {}).get("status") or "")),
                grammar_id=str((payload.get("decode_result") or {}).get("grammar_id") or ""),
                output=(
                    None
                    if (payload.get("decode_result") or {}).get("output") in (None, {})
                    else ResidualTaskOutput.from_dict((payload.get("decode_result") or {}).get("output"))
                ),
                reason_codes=tuple((payload.get("decode_result") or {}).get("reason_codes") or ()),
            ),
            task_output=ResidualTaskOutput.from_dict(payload.get("task_output") or {}),
            form=StructuredSpecialistForm(str(payload.get("form") or "")),
            disposition=ExpertDisposition(str(payload.get("disposition") or "")),
            feature_identity=str(payload.get("feature_identity") or ""),
            abstention=(
                None
                if abstention_payload in (None, {})
                else AbstentionDecision.from_dict(abstention_payload)
            ),
            independent_validator_identity=str(
                payload.get("independent_validator_identity") or ""
            ),
            structured_valid=payload.get("structured_valid"),
            model_calls=payload.get("model_calls", 0),
            provider_invocations=payload.get("provider_invocations", 0),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("prediction_id") or "")
        if claimed and claimed != result.prediction_id:
            raise ResidualIntelligenceError("structured specialist prediction identity mismatch")
        return result


@dataclass(frozen=True)
class ConstrainedStructuredExpert:
    """One family-bounded class-E constrained decoder for hole or patch sketches."""

    task_family: ResidualTaskFamily
    calibration_group: CalibrationGroup
    operator_vocabulary: tuple[str, ...] = ()
    operation_vocabulary: tuple[str, ...] = ()
    maximum_changed_lines: int = DEFAULT_PATCH_LINE_BOUND
    compiler_available: bool = True
    decoder_available: bool = True
    selective_policy: SelectivePredictionPolicy | None = None
    fitted: bool = False
    admission_id: str = ""
    checkpoint_count: int = 0
    expert_class: ExpertClass = ExpertClass.E
    schema: str = STRUCTURED_SPECIALIST_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "expert_version",
            "task_family",
            "calibration_group",
            "operator_vocabulary",
            "operation_vocabulary",
            "maximum_changed_lines",
            "compiler_available",
            "decoder_available",
            "selective_policy",
            "fitted",
            "admission_id",
            "checkpoint_count",
            "expert_class",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != STRUCTURED_SPECIALIST_SCHEMA:
            raise ResidualIntelligenceError("unsupported constrained structured expert schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        if self.task_family not in ADMITTED_STRUCTURED_FAMILIES:
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_FAMILY)
        object.__setattr__(self, "expert_class", parse_expert_class(self.expert_class))
        if self.expert_class is not ExpertClass.E:
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_CLASS)
        if EXPERT_CLASS_FORMS[self.expert_class] != (CONSTRAINED_DECODER_FORM,):
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_CLASS)
        if not isinstance(self.calibration_group, CalibrationGroup):
            raise ResidualIntelligenceError("structured specialist requires a typed calibration group")
        if self.calibration_group.family is not self.task_family:
            raise ResidualIntelligenceError(REASON_FAMILY_MISMATCH)
        family_spec = family_spec_for(self.task_family)
        expert_spec = expert_spec_for(self.task_family, ExpertClass.E)
        family_spec.reject_unsupported_risk(self.calibration_group.risk)
        expert_spec.reject_unsupported_risk(self.calibration_group.risk)
        if family_spec.semantic_kind != "structured":
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_FAMILY)
        object.__setattr__(
            self,
            "operator_vocabulary",
            _closed_vocabulary(
                self.operator_vocabulary,
                "operator_vocabulary",
                default=DEFAULT_HOLE_OPERATORS,
            ),
        )
        object.__setattr__(
            self,
            "operation_vocabulary",
            _closed_vocabulary(
                self.operation_vocabulary,
                "operation_vocabulary",
                default=DEFAULT_PATCH_OPERATIONS,
            ),
        )
        object.__setattr__(
            self,
            "maximum_changed_lines",
            bounded_int(
                self.maximum_changed_lines,
                "maximum_changed_lines",
                minimum=1,
                maximum=10_000,
            ),
        )
        object.__setattr__(
            self, "compiler_available", _require_bool(self.compiler_available, "compiler_available")
        )
        object.__setattr__(
            self, "decoder_available", _require_bool(self.decoder_available, "decoder_available")
        )
        if self.selective_policy is not None and not isinstance(
            self.selective_policy, SelectivePredictionPolicy
        ):
            raise ResidualIntelligenceError("selective_policy must be SelectivePredictionPolicy")
        object.__setattr__(self, "fitted", _require_bool(self.fitted, "fitted"))
        object.__setattr__(
            self,
            "admission_id",
            ""
            if self.admission_id in (None, "")
            else required_text(self.admission_id, "admission_id"),
        )
        object.__setattr__(
            self,
            "checkpoint_count",
            bounded_int(
                self.checkpoint_count,
                "checkpoint_count",
                minimum=0,
                maximum=MAX_SPECIALIST_CHECKPOINTS,
            ),
        )
        if self.fitted and not self.admission_id:
            raise ResidualIntelligenceError("fitted structured specialist requires an admission_id")
        if family_spec.emit_prose_by_default or expert_spec.emit_prose_by_default:
            raise ResidualIntelligenceError("structured specialist cannot emit prose by default")
        if expert_spec.authority_class != CANDIDATE_ONLY_AUTHORITY:
            raise ResidualIntelligenceError("structured specialist authority_class must be candidate_only")
        if expert_spec.grammar().grammar_id != grammar_for(self.task_family).grammar_id:
            raise ResidualIntelligenceError(REASON_GRAMMAR_MISMATCH)

    @property
    def expert_version(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def family_spec(self) -> ResidualTaskFamilySpec:
        return family_spec_for(self.task_family)

    @property
    def expert_spec(self) -> ResidualExpertSpec:
        return expert_spec_for(self.task_family, ExpertClass.E)

    @property
    def grammar(self) -> ExpertGrammar:
        return grammar_for(self.task_family)

    @property
    def candidate_only(self) -> bool:
        return True

    @property
    def form(self) -> str:
        return CONSTRAINED_DECODER_FORM

    def maximum_output_bytes(self) -> int:
        return min(self.grammar.maximum_output_bytes, self.expert_spec.maximum_output_bytes)

    def maximum_output_tokens(self) -> int:
        return self.expert_spec.maximum_output_tokens

    def _feature_identity(self, task_input: ResidualTaskInput) -> str:
        return canonical_id(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/residual-structured-feature-vector@1",
                "task_family": task_input.task_family.value,
                "compact_features": dict(task_input.compact_features),
            }
        )

    def _length_violation(self, raw: str | bytes) -> bool:
        return (
            _utf8_len(raw) > self.maximum_output_bytes()
            or _token_estimate(raw) > self.maximum_output_tokens()
        )

    def _validate_context(self, task_input: ResidualTaskInput) -> tuple[str, ...]:
        if task_input.task_family is not self.task_family:
            return (REASON_REJECT_INPUT, REASON_FAMILY_MISMATCH)
        try:
            self.family_spec.validate_task_input(task_input)
            self.expert_spec.reject_unsupported_risk(task_input.risk_class)
        except ResidualIntelligenceError as extra:
            message = str(extra)
            if REASON_TOKEN_LIMIT in message:
                return (REASON_REJECT_INPUT, REASON_TOKEN_LIMIT)
            if REASON_UNKNOWN_COMPACT_FEATURE in message:
                return (REASON_REJECT_INPUT, REASON_UNKNOWN_COMPACT_FEATURE)
            if REASON_MISSING_COMPACT_FEATURE in message:
                return (REASON_REJECT_INPUT, REASON_MISSING_COMPACT_FEATURE)
            if "unsupported" in message:
                return (REASON_REJECT_INPUT, REASON_UNSUPPORTED_FAMILY_RISK)
            return (REASON_REJECT_INPUT,)
        features = task_input.compact_features
        if any(_looks_like_private_body(str(key)) for key in features):
            return (REASON_REJECT_INPUT, REASON_INCOMPLETE_CONTEXT)
        if features.get("context_complete") is False:
            return (REASON_ABSTAIN_ESCALATE, REASON_INCOMPLETE_CONTEXT)
        if self.task_family is ResidualTaskFamily.PROCEDURE_HOLE_FILLING:
            if not self.compiler_available:
                return (REASON_COMPILER_UNAVAILABLE, REASON_ABSTAIN_ESCALATE)
            satisfied = features.get("procedure_preconditions_satisfied")
            if satisfied is False:
                return (REASON_PRECONDITIONS, REASON_ABSTAIN_ESCALATE)
        return ()

    def _closed_operator(self, token: str) -> str:
        if token not in self.operator_vocabulary:
            raise ResidualIntelligenceError("operator_id is outside the closed vocabulary")
        return token

    def _closed_operation(self, token: str) -> str:
        lowered = token.casefold()
        if lowered in FORBIDDEN_PATCH_OPERATIONS or "delete_test" in lowered:
            raise ResidualIntelligenceError(REASON_TEST_DELETION)
        if "weaken_validation" in lowered:
            raise ResidualIntelligenceError(REASON_VALIDATION_WEAKENING)
        if token not in self.operation_vocabulary:
            raise ResidualIntelligenceError("operations are outside the closed vocabulary")
        return token

    def _safe_relative_paths(self, values: Sequence[str], *, allowed: Sequence[str]) -> tuple[str, ...]:
        allowed_set = set(allowed)
        for item in values:
            path = PurePosixPath(item)
            if path.is_absolute() or item in {".", ".."} or ".." in path.parts:
                raise ResidualIntelligenceError(REASON_OUT_OF_SCOPE)
            if allowed_set and item not in allowed_set:
                raise ResidualIntelligenceError(REASON_OUT_OF_SCOPE)
        return tuple(values)

    def emit_constrained(self, task_input: ResidualTaskInput) -> str | None:
        """Compile one grammar-valid candidate from bounded compact features."""

        if not isinstance(task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("emit_constrained requires ResidualTaskInput")
        if self._validate_context(task_input):
            return None
        features = task_input.compact_features
        score = DEFAULT_SCORE_PPM
        evidence: tuple[str, ...] = ()
        try:
            if self.task_family is ResidualTaskFamily.PROCEDURE_HOLE_FILLING:
                hole_id = _reject_private_name(str(features.get("hole_id") or ""), "hole_id")
                operator = features.get("operation")
                if not isinstance(operator, str) or not operator:
                    return None
                payload = {
                    "hole_id": hole_id,
                    "operator_id": self._closed_operator(operator),
                    "argument_reference_ids": list(
                        _as_token_list(features.get("symbol_ids") or (), "symbol_ids")
                    ),
                    "precondition_reference_ids": list(
                        _as_token_list(features.get("procedure_root") or (), "procedure_root")
                    ),
                }
                output_class = "PROCEDURE_HOLE_RESOLUTION"
                evidence = (f"hole:{hole_id}",)
            else:
                allowed_paths = _as_token_list(
                    features.get("allowed_paths") or (), "allowed_paths", allow_empty=False
                )
                symbol_ids = _as_token_list(
                    features.get("symbol_ids") or (), "symbol_ids", allow_empty=False
                )
                operation = features.get("operation")
                if not isinstance(operation, str) or not operation:
                    return None
                line_bound = features.get("maximum_changed_lines", self.maximum_changed_lines)
                line_bound = bounded_int(
                    line_bound,
                    "maximum_changed_lines",
                    minimum=1,
                    maximum=self.maximum_changed_lines,
                )
                payload = {
                    "files": list(self._safe_relative_paths(allowed_paths, allowed=allowed_paths)),
                    "symbol_ids": list(symbol_ids),
                    "operations": [self._closed_operation(operation)],
                    "maximum_changed_lines": line_bound,
                    "validation_ids": list(
                        _as_token_list(features.get("validation_ids") or (), "validation_ids")
                    ),
                }
                output_class = "PATCH_SKETCH"
                evidence = tuple(f"path:{item}" for item in allowed_paths[:8])
        except ResidualIntelligenceError:
            return None
        if output_class not in task_input.allowed_outputs:
            return None
        reject_candidate_authority(payload)
        if _contains_forbidden_marker(payload):
            return None
        return _envelope(
            output_class=output_class,
            payload=payload,
            score_ppm=score,
            calibration_group=self.calibration_group.group_key,
            abstained=False,
            reason_codes=(),
            evidence_references=evidence,
        )

    def _post_parse(self, task_input: ResidualTaskInput, output: ResidualTaskOutput) -> None:
        if output.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if output.output_class not in task_input.allowed_outputs:
            raise ResidualIntelligenceError("output_class is outside allowed_outputs")
        if output.abstained:
            if output.structured_payload:
                raise ResidualIntelligenceError("abstention structured_payload must be empty")
            return
        payload = bounded_json_mapping(output.structured_payload, "structured_payload")
        reject_candidate_authority(payload)
        reject_secret_material(payload, noun="structured_payload")
        if _contains_forbidden_marker(payload):
            raise ResidualIntelligenceError(REASON_FREEFORM_AUTHORITY)
        features = task_input.compact_features
        if self.task_family is ResidualTaskFamily.PROCEDURE_HOLE_FILLING:
            hole_id = _reject_private_name(str(payload.get("hole_id") or ""), "hole_id")
            expected = str(features.get("hole_id") or "")
            if expected and hole_id != expected:
                raise ResidualIntelligenceError("hole_id is outside the bounded context")
            self._closed_operator(str(payload.get("operator_id") or ""))
            return
        files = _as_token_list(payload.get("files") or (), "files", allow_empty=False)
        allowed = _as_token_list(features.get("allowed_paths") or (), "allowed_paths")
        self._safe_relative_paths(files, allowed=allowed)
        symbols = _as_token_list(payload.get("symbol_ids") or (), "symbol_ids", allow_empty=False)
        expected_symbols = _as_token_list(features.get("symbol_ids") or (), "context symbol_ids")
        if expected_symbols and any(item not in set(expected_symbols) for item in symbols):
            raise ResidualIntelligenceError("symbol_ids are outside the bounded context")
        operations = _as_token_list(payload.get("operations") or (), "operations", allow_empty=False)
        for item in operations:
            self._closed_operation(item)
        line_bound = bounded_int(
            payload.get("maximum_changed_lines"),
            "maximum_changed_lines",
            minimum=1,
            maximum=self.maximum_changed_lines,
        )
        context_bound = features.get("maximum_changed_lines")
        if context_bound not in (None, "") and line_bound > bounded_int(
            context_bound, "context maximum_changed_lines", minimum=1, maximum=10_000
        ):
            raise ResidualIntelligenceError("maximum_changed_lines exceeds bounded context")

    def decode(self, request: StructuredDecodeRequest) -> StructuredDecodeResult:
        """Strict grammar parse plus family post-parse. Failures are invalid_output."""

        if not isinstance(request, StructuredDecodeRequest):
            raise ResidualIntelligenceError("decode requires StructuredDecodeRequest")
        grammar = self.grammar
        if request.grammar_id and request.grammar_id != grammar.grammar_id:
            return _invalid_decode(grammar)
        raw = request.raw_output
        if raw == "":
            return _invalid_decode(grammar)
        if self._length_violation(raw):
            return StructuredDecodeResult(
                status=DecodeStatus.INVALID_OUTPUT,
                grammar_id=grammar.grammar_id,
                output=None,
                reason_codes=(REASON_INVALID_OUTPUT, REASON_MAX_LENGTH),
            )
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            parsed = None
        if isinstance(parsed, Mapping) and _contains_forbidden_marker(parsed):
            return _invalid_decode(grammar)
        result = decode_structured_output(raw, grammar)
        if result.status is DecodeStatus.INVALID_OUTPUT:
            return result
        assert result.output is not None
        try:
            if request.task_input.task_family is not self.task_family:
                raise ResidualIntelligenceError(REASON_FAMILY_MISMATCH)
            self._post_parse(request.task_input, result.output)
        except ResidualIntelligenceError:
            return _invalid_decode(grammar)
        return result

    def _abstention_output(self, reasons: Sequence[str]) -> ResidualTaskOutput:
        return ResidualTaskOutput(
            output_class=ABSTAIN_OUTPUT_CLASS,
            structured_payload={},
            confidence_or_score=0,
            calibration_group=self.calibration_group.group_key,
            abstained=True,
            reason_codes=tuple(reasons) if reasons else (REASON_ABSTAIN_ESCALATE,),
            evidence_references=(),
            candidate_only=True,
        )

    def _structured_valid(self, output: ResidualTaskOutput) -> bool:
        envelope = _envelope(
            output_class=output.output_class,
            payload=output.structured_payload,
            score_ppm=output.confidence_or_score,
            calibration_group=output.calibration_group,
            abstained=output.abstained,
            reason_codes=output.reason_codes,
            evidence_references=output.evidence_references,
        )
        decoded = decode_structured_output(envelope, self.grammar)
        return decoded.status is DecodeStatus.VALID

    def predict(self, request: StructuredDecodeRequest) -> StructuredSpecialistPrediction:
        if not isinstance(request, StructuredDecodeRequest):
            raise ResidualIntelligenceError("predict requires StructuredDecodeRequest")
        task_input = request.task_input
        feature_identity = self._feature_identity(task_input)
        context_reasons = self._validate_context(task_input)
        if context_reasons:
            disposition = ExpertDisposition.REJECT_INPUT
            form = StructuredSpecialistForm.REJECT_INPUT
            if REASON_COMPILER_UNAVAILABLE in context_reasons:
                disposition = ExpertDisposition.CAPABILITY_UNAVAILABLE
                form = StructuredSpecialistForm.ABSTAIN
            elif REASON_REJECT_INPUT not in context_reasons:
                disposition = ExpertDisposition.ABSTAIN
                form = StructuredSpecialistForm.ABSTAIN
            output = self._abstention_output(context_reasons)
            return StructuredSpecialistPrediction(
                decode_result=_invalid_decode(self.grammar)
                if disposition is ExpertDisposition.REJECT_INPUT
                else StructuredDecodeResult(
                    status=DecodeStatus.VALID,
                    grammar_id=self.grammar.grammar_id,
                    output=output,
                    reason_codes=(),
                ),
                task_output=output,
                form=form,
                disposition=disposition,
                feature_identity=feature_identity,
                structured_valid=self._structured_valid(output),
                candidate_only=True,
            )
        if not self.decoder_available and not request.has_decoder_emission:
            output = self._abstention_output(
                (REASON_DECODER_UNAVAILABLE, REASON_ABSTAIN_ESCALATE)
            )
            return StructuredSpecialistPrediction(
                decode_result=StructuredDecodeResult(
                    status=DecodeStatus.VALID,
                    grammar_id=self.grammar.grammar_id,
                    output=output,
                    reason_codes=(),
                ),
                task_output=output,
                form=StructuredSpecialistForm.ABSTAIN,
                disposition=ExpertDisposition.CAPABILITY_UNAVAILABLE,
                feature_identity=feature_identity,
                structured_valid=self._structured_valid(output),
                candidate_only=True,
            )
        decode_request = request
        if not request.has_decoder_emission:
            emitted = self.emit_constrained(task_input)
            if emitted is None:
                output = self._abstention_output(
                    (REASON_ABSTAIN_ESCALATE, REASON_INCOMPLETE_CONTEXT)
                )
                return StructuredSpecialistPrediction(
                    decode_result=StructuredDecodeResult(
                        status=DecodeStatus.VALID,
                        grammar_id=self.grammar.grammar_id,
                        output=output,
                        reason_codes=(),
                    ),
                    task_output=output,
                    form=StructuredSpecialistForm.ABSTAIN,
                    disposition=ExpertDisposition.ABSTAIN,
                    feature_identity=feature_identity,
                    structured_valid=self._structured_valid(output),
                    candidate_only=True,
                )
            decode_request = StructuredDecodeRequest(
                task_input=task_input,
                raw_output=emitted,
                independent_validation=request.independent_validation,
                grammar_id=self.grammar.grammar_id,
                candidate_only=True,
            )
        decoded = self.decode(decode_request)
        if decoded.status is DecodeStatus.INVALID_OUTPUT:
            output = self._abstention_output(
                decoded.reason_codes or (REASON_INVALID_OUTPUT, REASON_ABSTAIN_ESCALATE)
            )
            return StructuredSpecialistPrediction(
                decode_result=decoded,
                task_output=output,
                form=StructuredSpecialistForm.ABSTAIN,
                disposition=ExpertDisposition.ABSTAIN,
                feature_identity=feature_identity,
                structured_valid=self._structured_valid(output),
                candidate_only=True,
            )
        assert decoded.output is not None
        candidate = decoded.output
        validator = request.independent_validation
        validator_id = ""
        validation_satisfied = False
        if validator is not None:
            validator_id = validator.validator_identity
            validation_satisfied = validator.accepted and not candidate.abstained
        if not candidate.abstained and not validator_id:
            # Non-abstained candidates remain proposals until an admitted validator checks them.
            validation_satisfied = False
        abstention: AbstentionDecision | None = None
        if self.selective_policy is not None:
            abstention = selectively_predict(
                self.selective_policy,
                SelectivePredictionRequest(
                    group=self.calibration_group,
                    score_ppm=candidate.confidence_or_score,
                    input_valid=True,
                    capability_available=self.compiler_available and self.decoder_available,
                    out_of_distribution=False,
                    validation_satisfied=validation_satisfied,
                    critical_boundary=False,
                ),
            )
            disposition = abstention.disposition
        elif candidate.abstained:
            disposition = ExpertDisposition.ABSTAIN
        else:
            disposition = ExpertDisposition.ABSTAIN
            candidate = self._abstention_output(
                (REASON_CURRENT_EVIDENCE, REASON_NO_GROUP_THRESHOLD)
            )
        if (
            disposition is ExpertDisposition.ACCEPT
            and (
                task_input.risk_class in PROPOSAL_RISKS
                or (validator is not None and not validator.accepted)
            )
        ):
            # R4/R5 stay proposal-tier; a rejected validator cannot ACCEPT.
            disposition = ExpertDisposition.VALIDATION_REQUIRED
        if disposition not in {ExpertDisposition.ACCEPT, ExpertDisposition.VALIDATION_REQUIRED}:
            if not candidate.abstained:
                candidate = self._abstention_output(
                    candidate.reason_codes or (REASON_ABSTAIN_ESCALATE,)
                )
            form = StructuredSpecialistForm.ABSTAIN
        else:
            form = StructuredSpecialistForm.CONSTRAINED_STRUCTURED_DECODER
            if disposition is ExpertDisposition.VALIDATION_REQUIRED:
                extra = () if abstention is None else abstention.reason_codes
                candidate = _keep_proposal_candidate(
                    candidate,
                    reasons=_validation_required_reasons(
                        candidate.reason_codes,
                        risk=task_input.risk_class,
                        extra=extra,
                    ),
                )
        if candidate.abstained and disposition in {
            ExpertDisposition.ACCEPT,
            ExpertDisposition.VALIDATION_REQUIRED,
        }:
            disposition = ExpertDisposition.ABSTAIN
            form = StructuredSpecialistForm.ABSTAIN
        return StructuredSpecialistPrediction(
            decode_result=decoded
            if form is StructuredSpecialistForm.CONSTRAINED_STRUCTURED_DECODER
            else (
                decoded
                if decoded.status is DecodeStatus.INVALID_OUTPUT
                else StructuredDecodeResult(
                    status=DecodeStatus.VALID,
                    grammar_id=self.grammar.grammar_id,
                    output=candidate,
                    reason_codes=(),
                )
            ),
            task_output=candidate,
            form=form,
            disposition=disposition,
            feature_identity=feature_identity,
            abstention=abstention,
            independent_validator_identity=validator_id,
            structured_valid=self._structured_valid(candidate),
            candidate_only=True,
        )

    def fit(
        self,
        *,
        admission: TrainingCorpusAdmission,
        examples: int = 1,
        steps: int = 1,
        wall_seconds: int = 0,
        gpu_seconds: int = 0,
    ) -> ConstrainedStructuredExpert:
        if not isinstance(admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("fit requires TrainingCorpusAdmission")
        if admission.admission_decision is not TrainingAvailability.ADMITTED:
            raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)
        admission.require_training_admitted()
        if not admission.can_train:
            raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)
        bounded_int(examples, "examples", minimum=1, maximum=MAX_SPECIALIST_EXAMPLES)
        bounded_int(steps, "steps", minimum=1, maximum=MAX_SPECIALIST_STEPS)
        bounded_int(wall_seconds, "wall_seconds", minimum=0, maximum=MAX_SPECIALIST_WALL_SECONDS)
        bounded_int(gpu_seconds, "gpu_seconds", minimum=0, maximum=MAX_SPECIALIST_GPU_SECONDS)
        if self.checkpoint_count >= MAX_SPECIALIST_CHECKPOINTS:
            raise ResidualIntelligenceError(
                f"structured specialist fit exceeds {MAX_SPECIALIST_CHECKPOINTS} checkpoints"
            )
        return ConstrainedStructuredExpert(
            schema=self.schema,
            task_family=self.task_family,
            calibration_group=self.calibration_group,
            operator_vocabulary=self.operator_vocabulary,
            operation_vocabulary=self.operation_vocabulary,
            maximum_changed_lines=self.maximum_changed_lines,
            compiler_available=self.compiler_available,
            decoder_available=self.decoder_available,
            selective_policy=self.selective_policy,
            fitted=True,
            admission_id=admission.admission_id,
            checkpoint_count=self.checkpoint_count + 1,
            expert_class=ExpertClass.E,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "calibration_group": self.calibration_group.to_dict(),
            "operator_vocabulary": list(self.operator_vocabulary),
            "operation_vocabulary": list(self.operation_vocabulary),
            "maximum_changed_lines": self.maximum_changed_lines,
            "compiler_available": self.compiler_available,
            "decoder_available": self.decoder_available,
            "selective_policy": (
                None if self.selective_policy is None else self.selective_policy.to_dict()
            ),
            "fitted": self.fitted,
            "admission_id": self.admission_id,
            "checkpoint_count": self.checkpoint_count,
            "expert_class": self.expert_class.value,
        }
        if include_id:
            result["expert_version"] = self.expert_version
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConstrainedStructuredExpert:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {
                "expert_version",
                "operator_vocabulary",
                "operation_vocabulary",
                "maximum_changed_lines",
                "compiler_available",
                "decoder_available",
                "selective_policy",
                "fitted",
                "admission_id",
                "checkpoint_count",
                "expert_class",
            },
            noun="constrained structured expert",
        )
        policy_payload = payload.get("selective_policy")
        group_payload = payload.get("calibration_group")
        if not isinstance(group_payload, Mapping):
            raise ResidualIntelligenceError("calibration_group must be an object")
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            calibration_group=CalibrationGroup.from_dict(group_payload),
            operator_vocabulary=tuple(payload.get("operator_vocabulary") or ()),
            operation_vocabulary=tuple(payload.get("operation_vocabulary") or ()),
            maximum_changed_lines=payload.get("maximum_changed_lines", DEFAULT_PATCH_LINE_BOUND),
            compiler_available=payload.get("compiler_available", True),
            decoder_available=payload.get("decoder_available", True),
            selective_policy=(
                None
                if policy_payload in (None, {})
                else SelectivePredictionPolicy.from_dict(policy_payload)
            ),
            fitted=payload.get("fitted", False),
            admission_id=str(payload.get("admission_id") or ""),
            checkpoint_count=payload.get("checkpoint_count", 0),
            expert_class=parse_expert_class(str(payload.get("expert_class") or ExpertClass.E.value)),
        )
        claimed = str(payload.get("expert_version") or "")
        if claimed and claimed != result.expert_version:
            raise ResidualIntelligenceError("constrained structured expert identity mismatch")
        return result


__all__ = (
    "ADMITTED_STRUCTURED_FAMILIES",
    "CONSTRAINED_DECODER_FORM",
    "ConstrainedStructuredExpert",
    "DecodeStatus",
    "DEFAULT_HOLE_OPERATORS",
    "DEFAULT_PATCH_OPERATIONS",
    "MAX_SPECIALIST_CHECKPOINTS",
    "MAX_SPECIALIST_EXAMPLES",
    "MAX_SPECIALIST_GPU_SECONDS",
    "MAX_SPECIALIST_STEPS",
    "MAX_SPECIALIST_WALL_SECONDS",
    "REASON_ABSTAIN_ESCALATE",
    "REASON_INVALID_OUTPUT",
    "REASON_MAX_LENGTH",
    "STRUCTURED_DECODE_REQUEST_SCHEMA",
    "STRUCTURED_SPECIALIST_SCHEMA",
    "StructuredDecodeRequest",
    "StructuredDecodeResult",
    "StructuredSpecialistForm",
    "StructuredSpecialistPrediction",
)
