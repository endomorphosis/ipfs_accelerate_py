"""Local exact, linear, and small-ranking residual experts.

The first local candidates cover admitted low/medium-risk classification and
ranking families.  Inference is provider-free and batched.  Accept still
requires grouped calibration, structured validity, and an independent
validator; larger forms need current held-out evidence of a routing-changing
gain.  Fitting is forbidden while training is unavailable.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .abstention import (
    AbstentionDecision,
    SelectivePredictionPolicy,
    SelectivePredictionRequest,
    selectively_predict,
)
from .baselines import (
    ABSTAIN_OUTPUT_CLASS,
    FEATURE_RANKING_CANDIDATES,
    FEATURE_RANKING_SIGNALS,
    LOGIT_SCALE,
    MAX_COEFFICIENT,
    MAX_FEATURE_VALUE,
    MAX_FEATURES,
    MAX_LOOKUP_ENTRIES,
    MAX_RANKING_ITEMS,
    MAX_RULES,
    DeclarativeRule,
    ExactLookupEntry,
    LinearForm,
    RankingItem,
    extract_linear_vector,
    logistic_ppm,
    stable_feature_identity,
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
    MIN_ROUTING_CHANGING_DELTA_PPM,
    ExpertClass,
    ResidualExpertSpec,
    admit_expert_class,
    expert_class_rank,
    expert_spec_for,
    parse_expert_class,
)
from .ood import (
    CANDIDATE_ONLY_AUTHORITY,
    BoundaryContract,
    OODAssessment,
    ReferenceDistribution,
    assess_out_of_distribution,
    observation_from_task_input,
)
from .residual_ir import ResidualIntelligenceIR, ResidualTaskInput, ResidualTaskOutput
from .rights import TrainingCorpusAdmission
from .splits import SplitPartition
from .structured_decoding import DecodeStatus, decode_structured_output, grammar_for
from .task_families import (
    REASON_UNSUPPORTED_FAMILY_RISK,
    SEMANTIC_KIND_CLASSIFICATION,
    SEMANTIC_KIND_RANKING,
    ResidualTaskFamilySpec,
    family_spec_for,
)

LOCAL_EXPERT_PREDICTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-local-expert-prediction@1"
)
LOCAL_EXPERT_COST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-local-expert-cost@1"
)
LOCAL_CLASSIFICATION_EXPERT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-local-classification-expert@1"
)
LOCAL_RANKING_EXPERT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-local-ranking-expert@1"
)
SMALL_RANKER_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-small-ranker@1"
BATCHED_EXPERT_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-batched-expert-request@1"
)
EXPERT_EVALUATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-local-expert-evaluation@1"
)
INDEPENDENT_VALIDATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-independent-validation-receipt@1"
)
LOCAL_FEATURE_VECTOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-local-feature-vector@1"
)

MAX_BATCH_ITEMS: Final = 1_024
MAX_LOCAL_EXAMPLES: Final = 50_000
MAX_LOCAL_STEPS: Final = 10_000
MAX_LOCAL_WALL_SECONDS: Final = 7_200
MAX_LOCAL_GPU_SECONDS: Final = 0
MAX_LOCAL_CHECKPOINTS: Final = 3
LOW_MEDIUM_RISKS: Final[frozenset[RiskClass]] = frozenset(
    {RiskClass.R0, RiskClass.R1, RiskClass.R2, RiskClass.R3}
)
PROPOSAL_RISKS: Final[frozenset[RiskClass]] = frozenset({RiskClass.R4, RiskClass.R5})

REASON_REJECT_INPUT: Final = "reject_input"
REASON_FAMILY_MISMATCH: Final = "task_family_mismatch"
REASON_UNSUPPORTED_KIND: Final = "unsupported_semantic_kind"
REASON_LOW_MEDIUM_RISK: Final = "local_experts_admit_low_medium_risk_only"
REASON_NO_DETERMINISTIC_MATCH: Final = "no_deterministic_match"
REASON_MISSING_STABLE_FEATURE: Final = "missing_stable_feature"
REASON_LINEAR_UNAVAILABLE: Final = "linear_coefficients_unavailable"
REASON_LINEAR_NO_SIGNAL: Final = "linear_no_signal"
REASON_RANKING_TIE: Final = "ranking_score_tie"
REASON_EMPTY_RANKING: Final = "empty_ranking_candidates"
REASON_SMALL_RANKER_UNAVAILABLE: Final = "small_ranker_unavailable"
REASON_INVALID_OUTPUT: Final = "invalid_output"
REASON_TRAINING_UNAVAILABLE: Final = "training_unavailable"
REASON_CURRENT_EVIDENCE: Final = "current_evidence_required"
REASON_NO_GROUP_THRESHOLD: Final = "no_group_threshold"
REASON_VALIDATION_REQUIRED: Final = "validation_required"
REASON_INDEPENDENT_VALIDATOR: Final = "independent_validator_required"
REASON_GPU_FORBIDDEN: Final = "local_experts_gpu_seconds_must_be_zero"
REASON_EXACT_LOOKUP: Final = "exact_lookup"
REASON_DETERMINISTIC_RULE: Final = "deterministic_rule"
REASON_DETERMINISTIC_RANKING: Final = "deterministic_ranking"
REASON_LINEAR_LOGISTIC: Final = "linear_logistic"
REASON_SMALL_RANKER: Final = "small_ranker"
REASON_OOD: Final = "out_of_distribution"

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


class LocalExpertForm(str, Enum):
    """Closed producer used for one local-expert prediction."""

    EXACT_LOOKUP = "exact_lookup"
    DECLARATIVE_RULE = "declarative_rule"
    DETERMINISTIC_RANKING = "deterministic_ranking"
    LINEAR_LOGISTIC = "linear_logistic"
    SMALL_RANKER = "small_ranker"
    ABSTAIN = "abstain"
    REJECT_INPUT = "reject_input"


FORM_EXPERT_CLASS: Final[Mapping[LocalExpertForm, ExpertClass]] = {
    LocalExpertForm.EXACT_LOOKUP: ExpertClass.A,
    LocalExpertForm.DECLARATIVE_RULE: ExpertClass.B,
    LocalExpertForm.DETERMINISTIC_RANKING: ExpertClass.B,
    LocalExpertForm.LINEAR_LOGISTIC: ExpertClass.C,
    LocalExpertForm.SMALL_RANKER: ExpertClass.D,
}

PRODUCER_FORMS: Final[frozenset[LocalExpertForm]] = frozenset(
    {
        LocalExpertForm.EXACT_LOOKUP,
        LocalExpertForm.DECLARATIVE_RULE,
        LocalExpertForm.DETERMINISTIC_RANKING,
        LocalExpertForm.LINEAR_LOGISTIC,
        LocalExpertForm.SMALL_RANKER,
    }
)


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _ppm(value: Any, name: str) -> int:
    return bounded_int(value, name, minimum=0, maximum=MAX_SCORE_PPM)


def _clamp_ppm(value: int) -> int:
    if value < 0:
        return 0
    if value > MAX_SCORE_PPM:
        return MAX_SCORE_PPM
    return value


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


def _reject_private_mapping(value: Mapping[str, Any], *, noun: str) -> dict[str, Any]:
    payload = bounded_json_mapping(value, noun)
    reject_secret_material(payload, noun=noun)
    reject_candidate_authority(payload)
    return payload


def _feature_names(values: Any) -> tuple[str, ...]:
    names = text_tuple(values, "feature_names", max_items=MAX_FEATURES)
    return tuple(_reject_private_name(item, "feature_names item") for item in names)


def _lookup_entries(values: Any) -> tuple[ExactLookupEntry, ...]:
    if values in (None, ()):
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError("lookup must be a sequence")
    if len(values) > MAX_LOOKUP_ENTRIES:
        raise ResidualIntelligenceError(f"lookup exceeds {MAX_LOOKUP_ENTRIES} entries")
    entries = tuple(
        item if isinstance(item, ExactLookupEntry) else ExactLookupEntry.from_dict(item)
        for item in values
    )
    keys = [item.feature_identity for item in entries]
    if len(set(keys)) != len(keys):
        raise ResidualIntelligenceError("exact lookup contains duplicate feature identities")
    return entries


def _rule_entries(values: Any) -> tuple[DeclarativeRule, ...]:
    if values in (None, ()):
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError("rules must be a sequence")
    if len(values) > MAX_RULES:
        raise ResidualIntelligenceError(f"rules exceed {MAX_RULES} entries")
    rules = tuple(
        item if isinstance(item, DeclarativeRule) else DeclarativeRule.from_dict(item)
        for item in values
    )
    ids = [item.rule_id for item in rules]
    if len(set(ids)) != len(ids):
        raise ResidualIntelligenceError("declarative rules contain duplicate rule_id values")
    return tuple(sorted(rules, key=lambda item: (-item.priority, item.rule_id)))


def _ranking_tuple(values: Any) -> tuple[RankingItem, ...]:
    if values in (None, ()):
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError("ranking must be a sequence")
    if len(values) > MAX_RANKING_ITEMS:
        raise ResidualIntelligenceError(f"ranking exceeds {MAX_RANKING_ITEMS} items")
    items: list[RankingItem] = []
    for item in values:
        if isinstance(item, RankingItem):
            items.append(item)
        elif isinstance(item, Mapping):
            items.append(RankingItem.from_dict(item))
        else:
            raise ResidualIntelligenceError("ranking items must be typed records")
    ids = [item.reference_id for item in items]
    if len(set(ids)) != len(ids):
        raise ResidualIntelligenceError("ranking contains duplicate reference identities")
    ordered = tuple(sorted(items, key=lambda item: (-item.score_ppm, item.reference_id)))
    if ordered != tuple(items):
        raise ResidualIntelligenceError("ranking is not stably ordered")
    return ordered


def _weight_pairs(values: Any) -> tuple[tuple[str, int], ...]:
    if values in (None, (), {}):
        return ()
    if isinstance(values, Mapping):
        pairs = tuple(
            (_reject_private_name(str(key), "ranking weight key"), _ppm(score, "ranking weight"))
            for key, score in values.items()
        )
    else:
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
            raise ResidualIntelligenceError("ranking_weights must be a sequence or object")
        pairs = []
        for item in values:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
                raise ResidualIntelligenceError("ranking weight pair must be a two-item sequence")
            if len(item) != 2:
                raise ResidualIntelligenceError("ranking weight pair must be a two-item sequence")
            pairs.append(
                (
                    _reject_private_name(str(item[0]), "ranking weight key"),
                    _ppm(item[1], "ranking weight"),
                )
            )
        pairs = tuple(pairs)
    keys = [item[0] for item in pairs]
    if len(set(keys)) != len(keys):
        raise ResidualIntelligenceError("ranking_weights contain duplicate keys")
    return tuple(sorted(pairs, key=lambda item: item[0]))


def _coefficient_rows(
    values: Any, *, n_classes: int, n_features: int
) -> tuple[tuple[int, ...], ...]:
    if values in (None, ()):
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError("coefficients must be a sequence of rows")
    if n_classes == 0:
        raise ResidualIntelligenceError("coefficients require class_labels")
    if len(values) != n_classes:
        raise ResidualIntelligenceError("coefficient rows must match class_labels")
    rows: list[tuple[int, ...]] = []
    for row in values:
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise ResidualIntelligenceError("coefficient row must be a sequence")
        if len(row) != n_features:
            raise ResidualIntelligenceError("coefficient row width must match feature_names")
        rows.append(
            tuple(
                bounded_int(
                    item,
                    "coefficient",
                    minimum=-MAX_COEFFICIENT,
                    maximum=MAX_COEFFICIENT,
                )
                for item in row
            )
        )
    return tuple(rows)


def _intercepts(values: Any, *, n_classes: int) -> tuple[int, ...]:
    if values in (None, ()):
        return tuple(0 for _ in range(n_classes))
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError("intercepts must be a sequence")
    if len(values) != n_classes:
        raise ResidualIntelligenceError("intercepts must match class_labels")
    return tuple(
        bounded_int(item, "intercept", minimum=-MAX_COEFFICIENT, maximum=MAX_COEFFICIENT)
        for item in values
    )


def _stable_local_value(value: Any, name: str) -> Any:
    if type(value) is bool:
        return value
    if type(value) is int:
        return bounded_int(value, name, minimum=-MAX_FEATURE_VALUE, maximum=MAX_FEATURE_VALUE)
    if isinstance(value, str):
        return _reject_private_name(value, name)
    if isinstance(value, Mapping):
        return {
            _reject_private_name(str(key), name): _stable_local_value(child, name)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_local_value(item, name) for item in value]
    raise ResidualIntelligenceError(f"{name} is not a stable local feature")


def extract_local_features(
    compact_features: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(compact_features, Mapping):
        raise ResidualIntelligenceError("compact_features must be an object")
    _reject_private_mapping(dict(compact_features), noun="compact_features")
    extracted: dict[str, Any] = {}
    for name in feature_names:
        if name in compact_features:
            extracted[name] = _stable_local_value(compact_features[name], name)
    return extracted


def local_feature_identity(
    compact_features: Mapping[str, Any],
    feature_names: Sequence[str],
) -> str:
    extracted = extract_local_features(compact_features, feature_names)
    return canonical_id(
        {
            "schema": LOCAL_FEATURE_VECTOR_SCHEMA,
            "feature_names": list(feature_names),
            "values": extracted,
        }
    )


def extract_local_linear_vector(
    compact_features: Mapping[str, Any],
    feature_names: Sequence[str],
) -> tuple[int, ...] | None:
    if any(name not in compact_features for name in feature_names):
        return None
    values: list[int] = []
    for name in feature_names:
        observed = compact_features[name]
        if type(observed) is bool:
            values.append(1 if observed else 0)
            continue
        if type(observed) is int:
            try:
                values.append(
                    bounded_int(
                        observed, name, minimum=0, maximum=MAX_FEATURE_VALUE
                    )
                )
            except ResidualIntelligenceError:
                return None
            continue
        if isinstance(observed, str):
            try:
                _reject_private_name(observed, name)
            except ResidualIntelligenceError:
                return None
            values.append(0)
            continue
        if isinstance(observed, Sequence) and not isinstance(observed, (str, bytes, bytearray)):
            values.append(min(len(observed), MAX_FEATURE_VALUE))
            continue
        return None
    return tuple(values)


def form_permitted(form: LocalExpertForm, admitted: ExpertClass) -> bool:
    if form not in FORM_EXPERT_CLASS:
        return form in {LocalExpertForm.ABSTAIN, LocalExpertForm.REJECT_INPUT}
    return expert_class_rank(FORM_EXPERT_CLASS[form]) <= expert_class_rank(admitted)


def require_training_admission(admission: TrainingCorpusAdmission) -> None:
    if not isinstance(admission, TrainingCorpusAdmission):
        raise ResidualIntelligenceError("fit requires TrainingCorpusAdmission")
    if admission.admission_decision is not TrainingAvailability.ADMITTED:
        raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)
    admission.require_training_admitted()
    if not admission.can_train:
        raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)


def classification_payload(
    family: ResidualTaskFamily,
    label: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(extra or {})
    references = list(payload.get("reference_ids") or [])
    if family is ResidualTaskFamily.FAILURE_ATTRIBUTION:
        return {
            "failure_class": str(payload.get("failure_class") or label),
            "recommended_action": str(
                payload.get("recommended_action") or "expand_context_reference"
            ),
            "reference_ids": references,
        }
    if family is ResidualTaskFamily.EFFECT_CLASSIFICATION:
        classes = payload.get("effect_classes")
        return {
            "effect_classes": list(classes) if classes else [label],
            "reference_ids": references,
        }
    if family is ResidualTaskFamily.CONTEXT_SUFFICIENCY:
        sufficient = payload.get("sufficient")
        if type(sufficient) is not bool:
            sufficient = True
        return {
            "sufficient": sufficient,
            "missing_reference_ids": list(payload.get("missing_reference_ids") or []),
            "reason_code": str(payload.get("reason_code") or "context_occupancy"),
        }
    if family is ResidualTaskFamily.CACHE_REUSE_CLASSIFICATION:
        reuse = payload.get("reuse")
        if type(reuse) is not bool:
            reuse = True
        return {
            "reuse": reuse,
            "dependency_reference_ids": list(payload.get("dependency_reference_ids") or []),
            "reason_code": str(payload.get("reason_code") or "cache_identity"),
        }
    if family is ResidualTaskFamily.RETRY_OR_ESCALATE:
        decision = str(payload.get("decision") or label)
        if decision not in {"retry", "escalate", "stop"}:
            decision = "retry"
        return {
            "decision": decision,
            "reason_code": str(payload.get("reason_code") or "attempt_disposition"),
            "reference_ids": references,
        }
    if family is ResidualTaskFamily.MERGE_CONFLICT_CLASSIFICATION:
        return {
            "conflict_class": str(payload.get("conflict_class") or label),
            "symbol_ids": list(payload.get("symbol_ids") or []),
            "reference_ids": references,
        }
    if family is ResidualTaskFamily.DOCUMENTATION_CLAIM_CLASSIFICATION:
        rewrite = payload.get("rewrite_required")
        if type(rewrite) is not bool:
            rewrite = False
        return {
            "claim_class": str(payload.get("claim_class") or label),
            "rewrite_required": rewrite,
            "evidence_reference_ids": list(
                payload.get("evidence_reference_ids") or payload.get("evidence_ids") or []
            ),
        }
    if family is ResidualTaskFamily.HUMAN_ESCALATION_CLASSIFICATION:
        escalate = payload.get("escalate")
        if type(escalate) is not bool:
            escalate = False
        return {
            "escalate": escalate,
            "reason_code": str(payload.get("reason_code") or "prior_disposition"),
        }
    return {"label": str(payload.get("label") or label), "reference_ids": references}


def ranking_payload(
    family: ResidualTaskFamily,
    ranking: Sequence[RankingItem],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(extra or {})
    ids = [item.reference_id for item in ranking]
    scores = [item.score_ppm for item in ranking]
    if family is ResidualTaskFamily.TEST_SELECTION:
        return {
            "test_ids": list(payload.get("test_ids") or ids),
            "coverage_reference_ids": list(payload.get("coverage_reference_ids") or []),
        }
    if family is ResidualTaskFamily.PROOF_SELECTION:
        return {
            "proof_ids": list(payload.get("proof_ids") or ids),
            "obligation_reference_ids": list(payload.get("obligation_reference_ids") or []),
        }
    return {
        "ranked_reference_ids": ids,
        "scores_ppm": scores,
    }


def structured_output_valid(output: ResidualTaskOutput, family: ResidualTaskFamily) -> bool:
    grammar = grammar_for(family)
    envelope = json.dumps(
        {
            "output_class": output.output_class,
            "structured_payload": dict(output.structured_payload),
            "confidence_or_score": output.confidence_or_score,
            "calibration_group": output.calibration_group,
            "abstained": output.abstained,
            "reason_codes": list(output.reason_codes),
            "evidence_references": list(output.evidence_references),
            "candidate_only": True,
        },
        separators=(",", ":"),
    )
    decoded = decode_structured_output(envelope, grammar)
    return decoded.status is DecodeStatus.VALID


def _rank_candidates(
    compact_features: Mapping[str, Any],
    ranking_weights: Mapping[str, int],
    *,
    scale: int = 1,
    extra_weights: Mapping[str, int] | None = None,
) -> tuple[RankingItem, ...]:
    raw = compact_features.get(FEATURE_RANKING_CANDIDATES)
    if raw in (None, ()):
        return ()
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise ResidualIntelligenceError("ranking_candidates must be a sequence of tokens")
    if len(raw) > MAX_RANKING_ITEMS:
        raise ResidualIntelligenceError(f"ranking_candidates exceed {MAX_RANKING_ITEMS} items")
    signals = compact_features.get(FEATURE_RANKING_SIGNALS) or {}
    if signals and not isinstance(signals, Mapping):
        raise ResidualIntelligenceError("ranking_signals must be an object")
    extras = extra_weights or {}
    items: list[RankingItem] = []
    seen: set[str] = set()
    for item in raw:
        reference = _reject_private_name(str(item), "ranking candidate")
        if reference in seen:
            raise ResidualIntelligenceError("ranking_candidates contain duplicates")
        seen.add(reference)
        signal = 0
        if isinstance(signals, Mapping) and reference in signals:
            signal = bounded_int(
                signals[reference],
                "ranking signal",
                minimum=0,
                maximum=MAX_SCORE_PPM,
            )
        score = _clamp_ppm(
            signal * scale + int(ranking_weights.get(reference, 0)) + int(extras.get(reference, 0))
        )
        items.append(RankingItem(reference_id=reference, score_ppm=score))
    return tuple(sorted(items, key=lambda item: (-item.score_ppm, item.reference_id)))


def _has_score_tie(ranking: Sequence[RankingItem]) -> bool:
    scores = [item.score_ppm for item in ranking]
    return len(scores) > 1 and len(set(scores)) != len(scores)


@dataclass(frozen=True)
class IndependentValidationReceipt:
    """Independent producer decision; never a model self-grade."""

    validator_identity: str
    accepted: bool
    evidence_references: tuple[str, ...] = ()
    candidate_only: bool = True
    schema: str = INDEPENDENT_VALIDATION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "receipt_id",
            "validator_identity",
            "accepted",
            "evidence_references",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != INDEPENDENT_VALIDATION_SCHEMA:
            raise ResidualIntelligenceError("unsupported independent validation schema")
        object.__setattr__(
            self,
            "validator_identity",
            required_text(self.validator_identity, "validator_identity"),
        )
        if not self.validator_identity.startswith("validator:"):
            raise ResidualIntelligenceError(REASON_INDEPENDENT_VALIDATOR)
        object.__setattr__(self, "accepted", _require_bool(self.accepted, "accepted"))
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references", max_items=256),
        )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("independent validation must remain candidate_only=true")

    @property
    def receipt_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "validator_identity": self.validator_identity,
            "accepted": self.accepted,
            "evidence_references": list(self.evidence_references),
            "candidate_only": True,
        }
        if include_id:
            result["receipt_id"] = self.receipt_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IndependentValidationReceipt:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"receipt_id", "evidence_references"},
            noun="independent validation receipt",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            validator_identity=str(payload.get("validator_identity") or ""),
            accepted=payload.get("accepted"),
            evidence_references=tuple(payload.get("evidence_references") or ()),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("receipt_id") or "")
        if claimed and claimed != result.receipt_id:
            raise ResidualIntelligenceError("independent validation identity mismatch")
        return result


@dataclass(frozen=True)
class LocalExpertCostReceipt:
    """Local-only cost record; remote and model fields are hard-zero."""

    form: LocalExpertForm
    feature_ops: int
    avoided_remote_calls: int
    avoided_strong_calls: int
    model_calls: int = 0
    provider_invocations: int = 0
    remote_input_tokens: int = 0
    remote_output_tokens: int = 0
    cost_microunits: int = 0
    latency_ms: int = 0
    schema: str = LOCAL_EXPERT_COST_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "receipt_id",
            "form",
            "feature_ops",
            "avoided_remote_calls",
            "avoided_strong_calls",
            "model_calls",
            "provider_invocations",
            "remote_input_tokens",
            "remote_output_tokens",
            "cost_microunits",
            "latency_ms",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != LOCAL_EXPERT_COST_SCHEMA:
            raise ResidualIntelligenceError("unsupported local expert cost schema")
        object.__setattr__(self, "form", LocalExpertForm(self.form))
        object.__setattr__(
            self,
            "feature_ops",
            bounded_int(self.feature_ops, "feature_ops", minimum=0, maximum=1_000_000),
        )
        for field in (
            "avoided_remote_calls",
            "avoided_strong_calls",
            "model_calls",
            "provider_invocations",
            "remote_input_tokens",
            "remote_output_tokens",
            "cost_microunits",
            "latency_ms",
        ):
            object.__setattr__(
                self,
                field,
                bounded_int(getattr(self, field), field, minimum=0, maximum=1_000_000_000_000),
            )
        if self.model_calls != 0 or self.provider_invocations != 0:
            raise ResidualIntelligenceError(
                "local expert cost receipts cannot record a model or provider call"
            )
        if self.remote_input_tokens != 0 or self.remote_output_tokens != 0:
            raise ResidualIntelligenceError(
                "local expert cost receipts cannot record remote tokens"
            )
        if self.avoided_remote_calls not in {0, 1} or self.avoided_strong_calls not in {0, 1}:
            raise ResidualIntelligenceError("per-prediction avoidance counts must be 0 or 1")

    @property
    def receipt_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def invoked_model_or_provider(self) -> bool:
        return False

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "form": self.form.value,
            "feature_ops": self.feature_ops,
            "avoided_remote_calls": self.avoided_remote_calls,
            "avoided_strong_calls": self.avoided_strong_calls,
            "model_calls": 0,
            "provider_invocations": 0,
            "remote_input_tokens": 0,
            "remote_output_tokens": 0,
            "cost_microunits": self.cost_microunits,
            "latency_ms": self.latency_ms,
        }
        if include_id:
            result["receipt_id"] = self.receipt_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LocalExpertCostReceipt:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"receipt_id"},
            noun="local expert cost receipt",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            form=LocalExpertForm(str(payload.get("form") or "")),
            feature_ops=payload.get("feature_ops"),
            avoided_remote_calls=payload.get("avoided_remote_calls"),
            avoided_strong_calls=payload.get("avoided_strong_calls"),
            model_calls=payload.get("model_calls"),
            provider_invocations=payload.get("provider_invocations"),
            remote_input_tokens=payload.get("remote_input_tokens"),
            remote_output_tokens=payload.get("remote_output_tokens"),
            cost_microunits=payload.get("cost_microunits"),
            latency_ms=payload.get("latency_ms"),
        )
        claimed = str(payload.get("receipt_id") or "")
        if claimed and claimed != result.receipt_id:
            raise ResidualIntelligenceError("local expert cost receipt identity mismatch")
        return result


def _local_cost(form: LocalExpertForm, *, feature_ops: int) -> LocalExpertCostReceipt:
    return LocalExpertCostReceipt(
        form=form,
        feature_ops=feature_ops,
        avoided_remote_calls=1,
        avoided_strong_calls=1,
    )


@dataclass(frozen=True)
class SmallRanker:
    """Bounded integer ranker over closed candidate identities."""

    candidate_weights: tuple[tuple[str, int], ...] = ()
    signal_scale: int = 1
    intercept_ppm: int = 0
    fitted: bool = False
    admission_id: str = ""
    checkpoint_count: int = 0
    schema: str = SMALL_RANKER_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "ranker_id",
            "candidate_weights",
            "signal_scale",
            "intercept_ppm",
            "fitted",
            "admission_id",
            "checkpoint_count",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != SMALL_RANKER_SCHEMA:
            raise ResidualIntelligenceError("unsupported small ranker schema")
        object.__setattr__(self, "candidate_weights", _weight_pairs(self.candidate_weights))
        object.__setattr__(
            self,
            "signal_scale",
            bounded_int(self.signal_scale, "signal_scale", minimum=0, maximum=MAX_SCORE_PPM),
        )
        object.__setattr__(self, "intercept_ppm", _ppm(self.intercept_ppm, "intercept_ppm"))
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
                maximum=MAX_LOCAL_CHECKPOINTS,
            ),
        )
        if self.fitted and not self.admission_id:
            raise ResidualIntelligenceError("fitted small ranker requires an admission_id")

    @property
    def ranker_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def available(self) -> bool:
        return bool(self.candidate_weights) or self.signal_scale > 0

    def weight_map(self) -> dict[str, int]:
        return {key: value for key, value in self.candidate_weights}

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "candidate_weights": [[key, value] for key, value in self.candidate_weights],
            "signal_scale": self.signal_scale,
            "intercept_ppm": self.intercept_ppm,
            "fitted": self.fitted,
            "admission_id": self.admission_id,
            "checkpoint_count": self.checkpoint_count,
        }
        if include_id:
            result["ranker_id"] = self.ranker_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SmallRanker:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {
                "ranker_id",
                "candidate_weights",
                "signal_scale",
                "intercept_ppm",
                "fitted",
                "admission_id",
                "checkpoint_count",
            },
            noun="small ranker",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            candidate_weights=tuple(payload.get("candidate_weights") or ()),
            signal_scale=payload.get("signal_scale", 1),
            intercept_ppm=payload.get("intercept_ppm", 0),
            fitted=payload.get("fitted", False),
            admission_id=str(payload.get("admission_id") or ""),
            checkpoint_count=payload.get("checkpoint_count", 0),
        )
        claimed = str(payload.get("ranker_id") or "")
        if claimed and claimed != result.ranker_id:
            raise ResidualIntelligenceError("small ranker identity mismatch")
        return result


@dataclass(frozen=True)
class LocalExpertPrediction:
    """Candidate-only local expert output with form, ranking, and receipts."""

    task_output: ResidualTaskOutput
    form: LocalExpertForm
    feature_identity: str
    cost: LocalExpertCostReceipt
    disposition: ExpertDisposition
    ranking: tuple[RankingItem, ...] = ()
    abstention: AbstentionDecision | None = None
    ood_assessment: OODAssessment | None = None
    structured_valid: bool = False
    independent_validator_identity: str = ""
    candidate_only: bool = True
    schema: str = LOCAL_EXPERT_PREDICTION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "prediction_id",
            "task_output",
            "form",
            "feature_identity",
            "cost",
            "disposition",
            "ranking",
            "abstention",
            "ood_assessment",
            "structured_valid",
            "independent_validator_identity",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != LOCAL_EXPERT_PREDICTION_SCHEMA:
            raise ResidualIntelligenceError("unsupported local expert prediction schema")
        if not isinstance(self.task_output, ResidualTaskOutput):
            raise ResidualIntelligenceError("task_output must be ResidualTaskOutput")
        if not isinstance(self.cost, LocalExpertCostReceipt):
            raise ResidualIntelligenceError("cost must be LocalExpertCostReceipt")
        object.__setattr__(self, "form", LocalExpertForm(self.form))
        object.__setattr__(self, "disposition", ExpertDisposition(self.disposition))
        object.__setattr__(
            self,
            "feature_identity",
            required_text(self.feature_identity, "feature_identity"),
        )
        object.__setattr__(self, "ranking", _ranking_tuple(self.ranking))
        if self.abstention is not None and not isinstance(self.abstention, AbstentionDecision):
            raise ResidualIntelligenceError("abstention must be AbstentionDecision")
        if self.ood_assessment is not None and not isinstance(self.ood_assessment, OODAssessment):
            raise ResidualIntelligenceError("ood_assessment must be OODAssessment")
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
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if self.task_output.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if self.cost.form is not self.form:
            raise ResidualIntelligenceError("cost receipt form must match the prediction form")
        if self.disposition is ExpertDisposition.ACCEPT:
            if self.task_output.abstained:
                raise ResidualIntelligenceError("ACCEPT cannot be abstained")
            if not self.structured_valid:
                raise ResidualIntelligenceError("ACCEPT requires structured validity")
            if not self.independent_validator_identity:
                raise ResidualIntelligenceError("ACCEPT requires an independent validator")
            if self.abstention is None or self.abstention.disposition is not ExpertDisposition.ACCEPT:
                raise ResidualIntelligenceError("ACCEPT requires a group-keyed abstention decision")
        if self.task_output.abstained and self.disposition is ExpertDisposition.ACCEPT:
            raise ResidualIntelligenceError("abstained output cannot ACCEPT")

    @property
    def prediction_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def as_ir(self, task_input: ResidualTaskInput) -> ResidualIntelligenceIR:
        return ResidualIntelligenceIR(task_input, self.task_output)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_output": self.task_output.to_dict(),
            "form": self.form.value,
            "feature_identity": self.feature_identity,
            "cost": self.cost.to_dict(),
            "disposition": self.disposition.value,
            "ranking": [item.to_dict() for item in self.ranking],
            "abstention": None if self.abstention is None else self.abstention.to_dict(),
            "ood_assessment": (
                None if self.ood_assessment is None else self.ood_assessment.to_dict()
            ),
            "structured_valid": self.structured_valid,
            "independent_validator_identity": self.independent_validator_identity,
            "candidate_only": True,
        }
        if include_id:
            result["prediction_id"] = self.prediction_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LocalExpertPrediction:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {
                "prediction_id",
                "ranking",
                "abstention",
                "ood_assessment",
                "independent_validator_identity",
            },
            noun="local expert prediction",
        )
        abstention_payload = payload.get("abstention")
        ood_payload = payload.get("ood_assessment")
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_output=ResidualTaskOutput.from_dict(payload.get("task_output") or {}),
            form=LocalExpertForm(str(payload.get("form") or "")),
            feature_identity=str(payload.get("feature_identity") or ""),
            cost=LocalExpertCostReceipt.from_dict(payload.get("cost") or {}),
            disposition=ExpertDisposition(str(payload.get("disposition") or "")),
            ranking=tuple(payload.get("ranking") or ()),
            abstention=(
                None
                if abstention_payload in (None, {})
                else AbstentionDecision.from_dict(abstention_payload)
            ),
            ood_assessment=(
                None
                if ood_payload in (None, {})
                else OODAssessment.from_dict(ood_payload)
            ),
            structured_valid=payload.get("structured_valid"),
            independent_validator_identity=str(
                payload.get("independent_validator_identity") or ""
            ),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("prediction_id") or "")
        if claimed and claimed != result.prediction_id:
            raise ResidualIntelligenceError("local expert prediction identity mismatch")
        return result


@dataclass(frozen=True)
class ExpertEvaluationCase:
    """One labelled local-expert evaluation row with complete identity."""

    task_input: ResidualTaskInput
    expected_output_class: str
    expected_payload: Mapping[str, Any] = None  # type: ignore[assignment]
    critical: bool = False
    adversarial: bool = False
    partition: SplitPartition = SplitPartition.HELD_OUT
    example_identity: str = ""
    independent_validation: IndependentValidationReceipt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("evaluation case requires ResidualTaskInput")
        object.__setattr__(
            self,
            "expected_output_class",
            required_text(self.expected_output_class, "expected_output_class"),
        )
        object.__setattr__(
            self,
            "expected_payload",
            _reject_private_mapping(self.expected_payload or {}, noun="expected_payload"),
        )
        object.__setattr__(self, "critical", _require_bool(self.critical, "critical"))
        object.__setattr__(self, "adversarial", _require_bool(self.adversarial, "adversarial"))
        object.__setattr__(self, "partition", SplitPartition(self.partition))
        object.__setattr__(
            self,
            "example_identity",
            ""
            if self.example_identity in (None, "")
            else required_text(self.example_identity, "example_identity"),
        )
        if self.independent_validation is not None and not isinstance(
            self.independent_validation, IndependentValidationReceipt
        ):
            raise ResidualIntelligenceError(
                "independent_validation must be IndependentValidationReceipt"
            )


@dataclass(frozen=True)
class ExpertEvaluation:
    """Held-out local-expert metrics with complete count denominators."""

    example_count: int
    held_out_count: int
    adversarial_count: int
    exact_lookup_count: int
    rule_count: int
    ranking_count: int
    linear_count: int
    small_ranker_count: int
    cascade_abstain_count: int
    reject_form_count: int
    accept_count: int
    abstain_count: int
    reject_input_count: int
    ood_count: int
    capability_unavailable_count: int
    validation_required_count: int
    false_accept_count: int
    critical_false_accept_count: int
    structured_valid_count: int
    structured_invalid_count: int
    independent_validator_accept_count: int
    independent_validator_reject_count: int
    avoided_model_calls: int
    avoided_remote_calls: int
    model_calls: int
    provider_invocations: int
    coverage_ppm: int
    precision_ppm: int
    abstention_rate_ppm: int
    group_key: str = ""
    schema: str = EXPERT_EVALUATION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "evaluation_id",
            "example_count",
            "held_out_count",
            "adversarial_count",
            "exact_lookup_count",
            "rule_count",
            "ranking_count",
            "linear_count",
            "small_ranker_count",
            "cascade_abstain_count",
            "reject_form_count",
            "accept_count",
            "abstain_count",
            "reject_input_count",
            "ood_count",
            "capability_unavailable_count",
            "validation_required_count",
            "false_accept_count",
            "critical_false_accept_count",
            "structured_valid_count",
            "structured_invalid_count",
            "independent_validator_accept_count",
            "independent_validator_reject_count",
            "avoided_model_calls",
            "avoided_remote_calls",
            "model_calls",
            "provider_invocations",
            "coverage_ppm",
            "precision_ppm",
            "abstention_rate_ppm",
            "group_key",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != EXPERT_EVALUATION_SCHEMA:
            raise ResidualIntelligenceError("unsupported local expert evaluation schema")
        counts: dict[str, int] = {}
        for field in (
            "example_count",
            "held_out_count",
            "adversarial_count",
            "exact_lookup_count",
            "rule_count",
            "ranking_count",
            "linear_count",
            "small_ranker_count",
            "cascade_abstain_count",
            "reject_form_count",
            "accept_count",
            "abstain_count",
            "reject_input_count",
            "ood_count",
            "capability_unavailable_count",
            "validation_required_count",
            "false_accept_count",
            "critical_false_accept_count",
            "structured_valid_count",
            "structured_invalid_count",
            "independent_validator_accept_count",
            "independent_validator_reject_count",
            "avoided_model_calls",
            "avoided_remote_calls",
            "model_calls",
            "provider_invocations",
        ):
            counts[field] = bounded_int(
                getattr(self, field),
                field,
                minimum=0,
                maximum=1_000_000_000_000,
            )
            object.__setattr__(self, field, counts[field])
        object.__setattr__(
            self,
            "group_key",
            "" if self.group_key in (None, "") else required_text(self.group_key, "group_key"),
        )
        n_examples = counts["example_count"]
        if n_examples < 1:
            raise ResidualIntelligenceError("evaluation requires at least one example")
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
                "disposition counts must equal the evaluation example population"
            )
        form_total = (
            counts["exact_lookup_count"]
            + counts["rule_count"]
            + counts["ranking_count"]
            + counts["linear_count"]
            + counts["small_ranker_count"]
            + counts["cascade_abstain_count"]
            + counts["reject_form_count"]
        )
        if form_total != n_examples:
            raise ResidualIntelligenceError(
                "form counts must equal the evaluation example population"
            )
        if counts["held_out_count"] + counts["adversarial_count"] > n_examples:
            raise ResidualIntelligenceError("held-out and adversarial counts exceed the population")
        if counts["false_accept_count"] > counts["accept_count"]:
            raise ResidualIntelligenceError("false accepts cannot exceed accepts")
        if counts["critical_false_accept_count"] > counts["false_accept_count"]:
            raise ResidualIntelligenceError("critical false accepts cannot exceed false accepts")
        if counts["model_calls"] != 0 or counts["provider_invocations"] != 0:
            raise ResidualIntelligenceError("local expert evaluation cannot include a model call")
        if counts["avoided_model_calls"] != n_examples:
            raise ResidualIntelligenceError("every local expert example must avoid a model call")
        if counts["avoided_remote_calls"] != n_examples:
            raise ResidualIntelligenceError("every local expert example must avoid a remote call")
        if (
            counts["structured_valid_count"] + counts["structured_invalid_count"]
            != n_examples
        ):
            raise ResidualIntelligenceError(
                "structured validity counts must equal the evaluation example population"
            )
        derived_coverage = (counts["accept_count"] * MAX_SCORE_PPM) // n_examples
        derived_precision = (
            0
            if counts["accept_count"] == 0
            else (
                (counts["accept_count"] - counts["false_accept_count"]) * MAX_SCORE_PPM
            )
            // counts["accept_count"]
        )
        derived_abstention = (counts["abstain_count"] * MAX_SCORE_PPM) // n_examples
        object.__setattr__(self, "coverage_ppm", _ppm(self.coverage_ppm, "coverage_ppm"))
        object.__setattr__(self, "precision_ppm", _ppm(self.precision_ppm, "precision_ppm"))
        object.__setattr__(
            self, "abstention_rate_ppm", _ppm(self.abstention_rate_ppm, "abstention_rate_ppm")
        )
        if self.coverage_ppm != derived_coverage:
            raise ResidualIntelligenceError("coverage_ppm does not match evaluation counts")
        if self.precision_ppm != derived_precision:
            raise ResidualIntelligenceError("precision_ppm does not match evaluation counts")
        if self.abstention_rate_ppm != derived_abstention:
            raise ResidualIntelligenceError(
                "abstention_rate_ppm does not match evaluation counts"
            )

    @property
    def evaluation_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def zero_critical_false_accepts(self) -> bool:
        return self.critical_false_accept_count == 0

    @property
    def quality_admitted(self) -> bool:
        return (
            self.zero_critical_false_accepts
            and self.structured_invalid_count == 0
            and self.model_calls == 0
            and self.held_out_count >= 1
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "example_count": self.example_count,
            "held_out_count": self.held_out_count,
            "adversarial_count": self.adversarial_count,
            "exact_lookup_count": self.exact_lookup_count,
            "rule_count": self.rule_count,
            "ranking_count": self.ranking_count,
            "linear_count": self.linear_count,
            "small_ranker_count": self.small_ranker_count,
            "cascade_abstain_count": self.cascade_abstain_count,
            "reject_form_count": self.reject_form_count,
            "accept_count": self.accept_count,
            "abstain_count": self.abstain_count,
            "reject_input_count": self.reject_input_count,
            "ood_count": self.ood_count,
            "capability_unavailable_count": self.capability_unavailable_count,
            "validation_required_count": self.validation_required_count,
            "false_accept_count": self.false_accept_count,
            "critical_false_accept_count": self.critical_false_accept_count,
            "structured_valid_count": self.structured_valid_count,
            "structured_invalid_count": self.structured_invalid_count,
            "independent_validator_accept_count": self.independent_validator_accept_count,
            "independent_validator_reject_count": self.independent_validator_reject_count,
            "avoided_model_calls": self.avoided_model_calls,
            "avoided_remote_calls": self.avoided_remote_calls,
            "model_calls": 0,
            "provider_invocations": 0,
            "coverage_ppm": self.coverage_ppm,
            "precision_ppm": self.precision_ppm,
            "abstention_rate_ppm": self.abstention_rate_ppm,
            "group_key": self.group_key,
        }
        if include_id:
            result["evaluation_id"] = self.evaluation_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExpertEvaluation:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"evaluation_id", "group_key"},
            noun="local expert evaluation",
        )
        kwargs = {
            key: payload.get(key)
            for key in cls._FIELDS
            if key not in {"evaluation_id", "schema"}
        }
        result = cls(schema=str(payload.get("schema") or ""), **kwargs)  # type: ignore[arg-type]
        claimed = str(payload.get("evaluation_id") or "")
        if claimed and claimed != result.evaluation_id:
            raise ResidualIntelligenceError("local expert evaluation identity mismatch")
        return result


def evaluate_local_predictions(
    cases: Sequence[ExpertEvaluationCase],
    predictions: Sequence[LocalExpertPrediction],
    *,
    group_key: str = "",
) -> ExpertEvaluation:
    if len(cases) != len(predictions):
        raise ResidualIntelligenceError("evaluation cases and predictions must align")
    if not cases:
        raise ResidualIntelligenceError("evaluation requires at least one example")
    exact = rule = ranking = linear = small = cascade_abstain = reject_form = 0
    accept = abstain = reject_input = ood = capability = validation_required = 0
    false_accept = critical_false_accept = 0
    structured_valid = structured_invalid = 0
    validator_accept = validator_reject = 0
    held_out = adversarial = 0
    for case, prediction in zip(cases, predictions):
        if prediction.cost.model_calls or prediction.cost.provider_invocations:
            raise ResidualIntelligenceError("local expert evaluation cannot include a model call")
        if prediction.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if case.partition is SplitPartition.HELD_OUT:
            held_out += 1
        if case.adversarial or case.partition is SplitPartition.ADVERSARIAL:
            adversarial += 1
        form = prediction.form
        if form is LocalExpertForm.EXACT_LOOKUP:
            exact += 1
        elif form is LocalExpertForm.DECLARATIVE_RULE:
            rule += 1
        elif form is LocalExpertForm.DETERMINISTIC_RANKING:
            ranking += 1
        elif form is LocalExpertForm.LINEAR_LOGISTIC:
            linear += 1
        elif form is LocalExpertForm.SMALL_RANKER:
            small += 1
        elif form is LocalExpertForm.REJECT_INPUT:
            reject_form += 1
        else:
            cascade_abstain += 1
        disposition = prediction.disposition
        if disposition is ExpertDisposition.ACCEPT:
            accept += 1
        elif disposition is ExpertDisposition.REJECT_INPUT:
            reject_input += 1
        elif disposition is ExpertDisposition.OUT_OF_DISTRIBUTION:
            ood += 1
        elif disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE:
            capability += 1
        elif disposition is ExpertDisposition.VALIDATION_REQUIRED:
            validation_required += 1
        else:
            abstain += 1
        if prediction.structured_valid:
            structured_valid += 1
        else:
            structured_invalid += 1
        receipt = case.independent_validation
        if receipt is not None and receipt.accepted:
            validator_accept += 1
        else:
            validator_reject += 1
        accepted = disposition is ExpertDisposition.ACCEPT
        wrong = prediction.task_output.output_class != case.expected_output_class
        if accepted and (wrong or case.critical or (receipt is not None and not receipt.accepted)):
            false_accept += 1
            if case.critical:
                critical_false_accept += 1
    n_examples = len(cases)
    return ExpertEvaluation(
        example_count=n_examples,
        held_out_count=held_out,
        adversarial_count=adversarial,
        exact_lookup_count=exact,
        rule_count=rule,
        ranking_count=ranking,
        linear_count=linear,
        small_ranker_count=small,
        cascade_abstain_count=cascade_abstain,
        reject_form_count=reject_form,
        accept_count=accept,
        abstain_count=abstain,
        reject_input_count=reject_input,
        ood_count=ood,
        capability_unavailable_count=capability,
        validation_required_count=validation_required,
        false_accept_count=false_accept,
        critical_false_accept_count=critical_false_accept,
        structured_valid_count=structured_valid,
        structured_invalid_count=structured_invalid,
        independent_validator_accept_count=validator_accept,
        independent_validator_reject_count=validator_reject,
        avoided_model_calls=n_examples,
        avoided_remote_calls=n_examples,
        model_calls=0,
        provider_invocations=0,
        coverage_ppm=(accept * MAX_SCORE_PPM) // n_examples,
        precision_ppm=(
            0 if accept == 0 else ((accept - false_accept) * MAX_SCORE_PPM) // accept
        ),
        abstention_rate_ppm=(abstain * MAX_SCORE_PPM) // n_examples,
        group_key=group_key,
    )


@dataclass(frozen=True)
class _CascadeResult:
    form: LocalExpertForm
    reasons: tuple[str, ...]
    output_class: str = ABSTAIN_OUTPUT_CLASS
    payload: Mapping[str, Any] | None = None
    score_ppm: int = 0
    evidence: tuple[str, ...] = ()
    ranking: tuple[RankingItem, ...] = ()


def _validate_family_and_risk(
    task_input: ResidualTaskInput,
    family_spec: ResidualTaskFamilySpec,
    expert_spec: ResidualExpertSpec,
) -> tuple[str, ...]:
    if task_input.task_family is not family_spec.task_family:
        return (REASON_REJECT_INPUT, REASON_FAMILY_MISMATCH)
    if task_input.task_family is not expert_spec.task_family:
        return (REASON_REJECT_INPUT, REASON_FAMILY_MISMATCH)
    try:
        family_spec.validate_task_input(task_input)
        expert_spec.reject_unsupported_risk(task_input.risk_class)
    except ResidualIntelligenceError as exc:
        message = str(exc)
        extra = REASON_UNSUPPORTED_FAMILY_RISK if "unsupported" in message else REASON_REJECT_INPUT
        return (REASON_REJECT_INPUT, extra)
    if task_input.risk_class not in LOW_MEDIUM_RISKS and task_input.risk_class not in PROPOSAL_RISKS:
        return (REASON_REJECT_INPUT, REASON_LOW_MEDIUM_RISK)
    if task_input.risk_class not in LOW_MEDIUM_RISKS and task_input.risk_class not in family_spec.allowed_risk_classes:
        return (REASON_REJECT_INPUT, REASON_LOW_MEDIUM_RISK)
    return ()


@dataclass(frozen=True)
class BatchedExpertRequest:
    """Ordered local inference batch; never a remote or simulated call."""

    task_inputs: tuple[ResidualTaskInput, ...]
    independent_validations: tuple[IndependentValidationReceipt | None, ...] = ()
    schema: str = BATCHED_EXPERT_REQUEST_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "request_id",
            "task_inputs",
            "independent_validations",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != BATCHED_EXPERT_REQUEST_SCHEMA:
            raise ResidualIntelligenceError("unsupported batched expert request schema")
        if isinstance(self.task_inputs, (str, bytes, bytearray)) or not isinstance(
            self.task_inputs, Sequence
        ):
            raise ResidualIntelligenceError("task_inputs must be a sequence")
        if not self.task_inputs:
            raise ResidualIntelligenceError("batched expert request requires at least one input")
        if len(self.task_inputs) > MAX_BATCH_ITEMS:
            raise ResidualIntelligenceError(f"batch exceeds {MAX_BATCH_ITEMS} items")
        if any(not isinstance(item, ResidualTaskInput) for item in self.task_inputs):
            raise ResidualIntelligenceError("batch items must be ResidualTaskInput")
        object.__setattr__(self, "task_inputs", tuple(self.task_inputs))
        validations = tuple(self.independent_validations)
        if validations and len(validations) != len(self.task_inputs):
            raise ResidualIntelligenceError(
                "independent_validations must align with task_inputs"
            )
        if any(
            item is not None and not isinstance(item, IndependentValidationReceipt)
            for item in validations
        ):
            raise ResidualIntelligenceError(
                "independent_validations must be typed receipts or null"
            )
        object.__setattr__(self, "independent_validations", validations)

    @property
    def request_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def receipt_for(self, index: int) -> IndependentValidationReceipt | None:
        if not self.independent_validations:
            return None
        return self.independent_validations[index]

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_inputs": [item.to_dict() for item in self.task_inputs],
            "independent_validations": [
                None if item is None else item.to_dict() for item in self.independent_validations
            ],
        }
        if include_id:
            result["request_id"] = self.request_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BatchedExpertRequest:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"request_id", "independent_validations"},
            noun="batched expert request",
        )
        validations_payload = payload.get("independent_validations") or ()
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_inputs=tuple(
                ResidualTaskInput.from_dict(item) for item in (payload.get("task_inputs") or ())
            ),
            independent_validations=tuple(
                None if item in (None, {}) else IndependentValidationReceipt.from_dict(item)
                for item in validations_payload
            ),
        )
        claimed = str(payload.get("request_id") or "")
        if claimed and claimed != result.request_id:
            raise ResidualIntelligenceError("batched expert request identity mismatch")
        return result


class _LocalExpertMixin:
    task_family: ResidualTaskFamily
    expert_class: ExpertClass
    calibration_group: CalibrationGroup
    feature_names: tuple[str, ...]
    lookup: tuple[ExactLookupEntry, ...]
    selective_policy: SelectivePredictionPolicy | None
    ood_reference: ReferenceDistribution | None
    ood_boundary: BoundaryContract | None
    policy_admits_ood: bool
    ood_schema: str
    ood_operation: str
    ood_effects: tuple[str, ...]
    ood_capabilities: tuple[str, ...]
    ood_context_fields: tuple[str, ...]

    @property
    def family_spec(self) -> ResidualTaskFamilySpec:
        return family_spec_for(self.task_family)

    @property
    def expert_spec(self) -> ResidualExpertSpec:
        return expert_spec_for(self.task_family, self.expert_class)

    @property
    def candidate_only(self) -> bool:
        return True

    def lookup_by_identity(self) -> dict[str, ExactLookupEntry]:
        return {item.feature_identity: item for item in self.lookup}

    def _feature_identity(self, task_input: ResidualTaskInput) -> str:
        try:
            return local_feature_identity(task_input.compact_features, self.feature_names)
        except ResidualIntelligenceError:
            return stable_feature_identity(task_input.compact_features, self.feature_names)

    def _assess_ood(self, task_input: ResidualTaskInput) -> OODAssessment | None:
        if self.ood_reference is None and self.ood_boundary is None:
            return None
        context = self.ood_context_fields or (
            (task_input.context_capsule_cid,) if task_input.context_capsule_cid else ()
        )
        complete = True
        if "context_complete" in task_input.compact_features:
            complete = task_input.compact_features["context_complete"] is True
        observation = observation_from_task_input(
            task_input,
            schema=self.ood_schema or self.family_spec.input_schema,
            operation=self.ood_operation,
            repository=self.calibration_group.repository,
            effects=self.ood_effects,
            authority_class=CANDIDATE_ONLY_AUTHORITY,
            calibration_group_key=self.calibration_group.group_key,
            context_fields=context,
            capabilities=self.ood_capabilities or self.expert_spec.capabilities,
            capability_available=True,
            detection_available=self.ood_reference is not None,
            context_complete=complete,
        )
        return assess_out_of_distribution(
            observation,
            reference=self.ood_reference,
            boundary=self.ood_boundary,
            policy_admits_ood=self.policy_admits_ood,
        )

    def _finalize(
        self,
        task_input: ResidualTaskInput,
        cascade: _CascadeResult,
        *,
        feature_identity: str,
        feature_ops: int,
        independent_validation: IndependentValidationReceipt | None,
    ) -> LocalExpertPrediction:
        ood = self._assess_ood(task_input)
        form = cascade.form
        reasons = cascade.reasons
        output_class = cascade.output_class
        payload = dict(cascade.payload or {})
        score_ppm = cascade.score_ppm
        evidence = cascade.evidence
        ranking = cascade.ranking
        abstained = form in {LocalExpertForm.ABSTAIN, LocalExpertForm.REJECT_INPUT}
        if ood is not None and ood.forced_disposition is not None:
            forced = ood.forced_disposition
            if forced is ExpertDisposition.OUT_OF_DISTRIBUTION:
                abstained = True
                form = LocalExpertForm.ABSTAIN if form is not LocalExpertForm.REJECT_INPUT else form
                output_class = ABSTAIN_OUTPUT_CLASS
                payload = {}
                score_ppm = 0
                evidence = ()
                ranking = ()
                reasons = (REASON_OOD,) + ood.reason_codes
            elif forced is ExpertDisposition.CAPABILITY_UNAVAILABLE:
                abstained = True
                form = LocalExpertForm.ABSTAIN if form is not LocalExpertForm.REJECT_INPUT else form
                output_class = ABSTAIN_OUTPUT_CLASS
                payload = {}
                score_ppm = 0
                evidence = ()
                ranking = ()
                reasons = ood.reason_codes
            elif forced is ExpertDisposition.REJECT_INPUT:
                form = LocalExpertForm.REJECT_INPUT
                abstained = True
                output_class = ABSTAIN_OUTPUT_CLASS
                payload = {}
                score_ppm = 0
                evidence = ()
                ranking = ()
                reasons = (REASON_REJECT_INPUT,) + ood.reason_codes
        if output_class not in task_input.allowed_outputs:
            if ABSTAIN_OUTPUT_CLASS not in task_input.allowed_outputs:
                raise ResidualIntelligenceError("ABSTAIN must be in allowed_outputs")
            abstained = True
            form = LocalExpertForm.ABSTAIN if form is not LocalExpertForm.REJECT_INPUT else form
            output_class = ABSTAIN_OUTPUT_CLASS
            payload = {}
            score_ppm = 0
            evidence = ()
            reasons = ("output_class_not_allowed",) + reasons
        if abstained:
            output_class = ABSTAIN_OUTPUT_CLASS
            payload = {}
            score_ppm = 0
            evidence = ()
        output = ResidualTaskOutput(
            output_class=output_class,
            structured_payload=payload,
            confidence_or_score=_ppm(score_ppm, "confidence_or_score"),
            calibration_group=self.calibration_group.group_key,
            abstained=abstained,
            reason_codes=reasons,
            evidence_references=evidence,
            candidate_only=True,
        )
        structured_valid = structured_output_valid(output, task_input.task_family)
        if not structured_valid:
            output = ResidualTaskOutput(
                output_class=ABSTAIN_OUTPUT_CLASS,
                structured_payload={},
                confidence_or_score=0,
                calibration_group=self.calibration_group.group_key,
                abstained=True,
                reason_codes=(REASON_INVALID_OUTPUT,) + reasons,
                evidence_references=(),
                candidate_only=True,
            )
            structured_valid = structured_output_valid(output, task_input.task_family)
            abstained = True
            if form is not LocalExpertForm.REJECT_INPUT:
                form = LocalExpertForm.ABSTAIN
            ranking = ()
        validator_id = ""
        validation_satisfied = False
        if independent_validation is not None:
            validator_id = independent_validation.validator_identity
            validation_satisfied = (
                structured_valid
                and independent_validation.accepted
                and not output.abstained
            )
        bound_ood = bool(ood is not None and ood.bound_ood)
        conservative = bool(ood is not None and ood.conservative_abstain)
        out_of_distribution = bound_ood or conservative
        abstention: AbstentionDecision | None = None
        if self.selective_policy is not None and form is not LocalExpertForm.REJECT_INPUT:
            request = SelectivePredictionRequest(
                group=self.calibration_group,
                score_ppm=output.confidence_or_score,
                input_valid=True,
                capability_available=not (
                    ood is not None
                    and ood.forced_disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE
                ),
                out_of_distribution=out_of_distribution,
                validation_satisfied=validation_satisfied,
                critical_boundary=conservative,
            )
            abstention = selectively_predict(self.selective_policy, request)
        if form is LocalExpertForm.REJECT_INPUT:
            disposition = ExpertDisposition.REJECT_INPUT
        elif abstention is not None:
            disposition = abstention.disposition
        elif output.abstained:
            if ood is not None and ood.forced_disposition is not None:
                disposition = ood.forced_disposition
            else:
                disposition = ExpertDisposition.ABSTAIN
        else:
            disposition = ExpertDisposition.ABSTAIN
            output = ResidualTaskOutput(
                output_class=ABSTAIN_OUTPUT_CLASS,
                structured_payload={},
                confidence_or_score=0,
                calibration_group=self.calibration_group.group_key,
                abstained=True,
                reason_codes=(REASON_CURRENT_EVIDENCE, REASON_NO_GROUP_THRESHOLD)
                + output.reason_codes,
                evidence_references=(),
                candidate_only=True,
            )
            structured_valid = structured_output_valid(output, task_input.task_family)
            ranking = ()
        if output.abstained and disposition in {
            ExpertDisposition.ACCEPT,
            ExpertDisposition.VALIDATION_REQUIRED,
        }:
            if ood is not None and ood.forced_disposition is not None:
                disposition = ood.forced_disposition
            else:
                disposition = ExpertDisposition.ABSTAIN
        if disposition not in {ExpertDisposition.ACCEPT, ExpertDisposition.VALIDATION_REQUIRED}:
            if not output.abstained:
                extra_reasons = output.reason_codes
                if disposition is ExpertDisposition.OUT_OF_DISTRIBUTION:
                    extra_reasons = (REASON_OOD,) + extra_reasons
                output = ResidualTaskOutput(
                    output_class=ABSTAIN_OUTPUT_CLASS,
                    structured_payload={},
                    confidence_or_score=0,
                    calibration_group=self.calibration_group.group_key,
                    abstained=True,
                    reason_codes=extra_reasons if extra_reasons else (REASON_CURRENT_EVIDENCE,),
                    evidence_references=(),
                    candidate_only=True,
                )
                structured_valid = structured_output_valid(output, task_input.task_family)
                ranking = ()
        elif disposition is ExpertDisposition.VALIDATION_REQUIRED and output.abstained:
            disposition = ExpertDisposition.ABSTAIN
        if (
            disposition is ExpertDisposition.ACCEPT
            and independent_validation is not None
            and not independent_validation.accepted
        ):
            disposition = ExpertDisposition.VALIDATION_REQUIRED
        return LocalExpertPrediction(
            task_output=output,
            form=form,
            feature_identity=feature_identity,
            cost=_local_cost(form, feature_ops=feature_ops),
            disposition=disposition,
            ranking=ranking,
            abstention=abstention,
            ood_assessment=ood,
            structured_valid=structured_valid,
            independent_validator_identity=validator_id,
            candidate_only=True,
        )


def _bind_local_expert(
    *,
    task_family: ResidualTaskFamily,
    expert_class: ExpertClass,
    calibration_group: CalibrationGroup,
    expected_kind: str,
) -> tuple[ResidualTaskFamilySpec, ResidualExpertSpec]:
    if not isinstance(calibration_group, CalibrationGroup):
        raise ResidualIntelligenceError("local experts require a typed calibration group")
    family_spec = family_spec_for(task_family)
    if family_spec.semantic_kind != expected_kind:
        raise ResidualIntelligenceError(REASON_UNSUPPORTED_KIND)
    wanted = parse_expert_class(expert_class)
    spec = expert_spec_for(task_family, wanted)
    spec.reject_unsupported_risk(calibration_group.risk)
    if calibration_group.family is not task_family:
        raise ResidualIntelligenceError(REASON_FAMILY_MISMATCH)
    if calibration_group.risk not in LOW_MEDIUM_RISKS and calibration_group.risk not in family_spec.allowed_risk_classes:
        raise ResidualIntelligenceError(REASON_LOW_MEDIUM_RISK)
    return family_spec, spec


@dataclass(frozen=True)
class LocalClassificationExpert(_LocalExpertMixin):
    """Exact/linear classification candidate for one low/medium-risk family."""

    task_family: ResidualTaskFamily
    expert_class: ExpertClass
    calibration_group: CalibrationGroup
    feature_names: tuple[str, ...]
    lookup: tuple[ExactLookupEntry, ...] = ()
    rules: tuple[DeclarativeRule, ...] = ()
    class_labels: tuple[str, ...] = ()
    linear_form: LinearForm = LinearForm.LOGISTIC
    coefficients: tuple[tuple[int, ...], ...] = ()
    intercepts: tuple[int, ...] = ()
    linear_threshold_ppm: int = 500_000
    selective_policy: SelectivePredictionPolicy | None = None
    ood_reference: ReferenceDistribution | None = None
    ood_boundary: BoundaryContract | None = None
    policy_admits_ood: bool = False
    ood_schema: str = ""
    ood_operation: str = ""
    ood_effects: tuple[str, ...] = ()
    ood_capabilities: tuple[str, ...] = ()
    ood_context_fields: tuple[str, ...] = ()
    fitted: bool = False
    admission_id: str = ""
    checkpoint_count: int = 0
    schema: str = LOCAL_CLASSIFICATION_EXPERT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "expert_version",
            "task_family",
            "expert_class",
            "calibration_group",
            "feature_names",
            "lookup",
            "rules",
            "class_labels",
            "linear_form",
            "coefficients",
            "intercepts",
            "linear_threshold_ppm",
            "selective_policy",
            "ood_reference",
            "ood_boundary",
            "policy_admits_ood",
            "ood_schema",
            "ood_operation",
            "ood_effects",
            "ood_capabilities",
            "ood_context_fields",
            "fitted",
            "admission_id",
            "checkpoint_count",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != LOCAL_CLASSIFICATION_EXPERT_SCHEMA:
            raise ResidualIntelligenceError("unsupported local classification expert schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "expert_class", parse_expert_class(self.expert_class))
        family_spec, spec = _bind_local_expert(
            task_family=self.task_family,
            expert_class=self.expert_class,
            calibration_group=self.calibration_group,
            expected_kind=SEMANTIC_KIND_CLASSIFICATION,
        )
        if spec.expert_class is not self.expert_class:
            raise ResidualIntelligenceError("classification expert class mismatch")
        if family_spec.semantic_kind != SEMANTIC_KIND_CLASSIFICATION:
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_KIND)
        object.__setattr__(self, "feature_names", _feature_names(self.feature_names))
        object.__setattr__(self, "lookup", _lookup_entries(self.lookup))
        object.__setattr__(self, "rules", _rule_entries(self.rules))
        labels = text_tuple(self.class_labels, "class_labels", max_items=256)
        if any(item == ABSTAIN_OUTPUT_CLASS for item in labels):
            raise ResidualIntelligenceError("class_labels cannot include ABSTAIN")
        object.__setattr__(self, "class_labels", labels)
        object.__setattr__(self, "linear_form", LinearForm(self.linear_form))
        coefficients = (
            _coefficient_rows(
                self.coefficients,
                n_classes=len(labels),
                n_features=len(self.feature_names),
            )
            if labels
            else ()
        )
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "intercepts",
            _intercepts(self.intercepts, n_classes=len(labels)) if coefficients else (),
        )
        object.__setattr__(
            self, "linear_threshold_ppm", _ppm(self.linear_threshold_ppm, "linear_threshold_ppm")
        )
        if self.selective_policy is not None and not isinstance(
            self.selective_policy, SelectivePredictionPolicy
        ):
            raise ResidualIntelligenceError("selective_policy must be SelectivePredictionPolicy")
        if self.ood_reference is not None and not isinstance(
            self.ood_reference, ReferenceDistribution
        ):
            raise ResidualIntelligenceError("ood_reference must be ReferenceDistribution")
        if self.ood_boundary is not None and not isinstance(self.ood_boundary, BoundaryContract):
            raise ResidualIntelligenceError("ood_boundary must be BoundaryContract")
        object.__setattr__(
            self, "policy_admits_ood", _require_bool(self.policy_admits_ood, "policy_admits_ood")
        )
        object.__setattr__(
            self,
            "ood_schema",
            "" if self.ood_schema in (None, "") else required_text(self.ood_schema, "ood_schema"),
        )
        object.__setattr__(
            self,
            "ood_operation",
            ""
            if self.ood_operation in (None, "")
            else required_text(self.ood_operation, "ood_operation"),
        )
        object.__setattr__(
            self, "ood_effects", text_tuple(self.ood_effects, "ood_effects")
        )
        object.__setattr__(
            self,
            "ood_capabilities",
            text_tuple(self.ood_capabilities, "ood_capabilities"),
        )
        object.__setattr__(
            self,
            "ood_context_fields",
            text_tuple(self.ood_context_fields, "ood_context_fields"),
        )
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
                maximum=MAX_LOCAL_CHECKPOINTS,
            ),
        )
        if self.fitted and not self.admission_id:
            raise ResidualIntelligenceError("fitted local expert requires an admission_id")

    @property
    def expert_version(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def coefficients_available(self) -> bool:
        return bool(self.class_labels) and bool(self.coefficients)

    def score_vector(self, values: Sequence[int]) -> tuple[int, ...]:
        if len(values) != len(self.feature_names):
            raise ResidualIntelligenceError("linear vector width mismatch")
        scores: list[int] = []
        for intercept, row in zip(self.intercepts, self.coefficients):
            raw = intercept + sum(coef * feature for coef, feature in zip(row, values))
            if self.linear_form is LinearForm.LOGISTIC:
                scores.append(logistic_ppm(raw))
            else:
                scores.append(_clamp_ppm(raw))
        return tuple(scores)

    def _cascade(self, task_input: ResidualTaskInput) -> _CascadeResult:
        rejected = _validate_family_and_risk(task_input, self.family_spec, self.expert_spec)
        if rejected:
            return _CascadeResult(form=LocalExpertForm.REJECT_INPUT, reasons=rejected)
        features = dict(task_input.compact_features)
        identity = self._feature_identity(task_input)
        if form_permitted(LocalExpertForm.EXACT_LOOKUP, self.expert_class):
            hit = self.lookup_by_identity().get(identity)
            if hit is not None:
                return _CascadeResult(
                    form=LocalExpertForm.EXACT_LOOKUP,
                    reasons=(REASON_EXACT_LOOKUP,),
                    output_class=hit.output_class,
                    payload=hit.structured_payload,
                    score_ppm=hit.score_ppm,
                    evidence=hit.evidence_references,
                )
        if form_permitted(LocalExpertForm.DECLARATIVE_RULE, self.expert_class):
            for rule in self.rules:
                if rule.matches(features):
                    return _CascadeResult(
                        form=LocalExpertForm.DECLARATIVE_RULE,
                        reasons=(REASON_DETERMINISTIC_RULE,),
                        output_class=rule.output_class,
                        payload=rule.structured_payload,
                        score_ppm=rule.score_ppm,
                        evidence=rule.evidence_references + (f"rule:{rule.rule_id}",),
                    )
        if form_permitted(LocalExpertForm.LINEAR_LOGISTIC, self.expert_class):
            if not self.coefficients_available:
                return _CascadeResult(
                    form=LocalExpertForm.ABSTAIN,
                    reasons=(REASON_LINEAR_UNAVAILABLE, REASON_NO_DETERMINISTIC_MATCH),
                )
            vector = extract_local_linear_vector(features, self.feature_names)
            if vector is None:
                vector = extract_linear_vector(features, self.feature_names)
            if vector is None:
                return _CascadeResult(
                    form=LocalExpertForm.ABSTAIN,
                    reasons=(REASON_MISSING_STABLE_FEATURE,),
                )
            scores = self.score_vector(vector)
            ranked = tuple(
                RankingItem(reference_id=label, score_ppm=score)
                for label, score in sorted(
                    zip(self.class_labels, scores),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            if not ranked:
                return _CascadeResult(
                    form=LocalExpertForm.ABSTAIN,
                    reasons=(REASON_LINEAR_NO_SIGNAL,),
                )
            best = ranked[0]
            if len(self.class_labels) == 1 and best.score_ppm < self.linear_threshold_ppm:
                return _CascadeResult(
                    form=LocalExpertForm.ABSTAIN,
                    reasons=(REASON_LINEAR_NO_SIGNAL,),
                    ranking=ranked,
                )
            if best.score_ppm <= 0 and self.linear_form is LinearForm.LINEAR:
                return _CascadeResult(
                    form=LocalExpertForm.ABSTAIN,
                    reasons=(REASON_LINEAR_NO_SIGNAL,),
                    ranking=ranked,
                )
            return _CascadeResult(
                form=LocalExpertForm.LINEAR_LOGISTIC,
                reasons=(REASON_LINEAR_LOGISTIC,),
                output_class=task_input.task_family.value,
                payload=classification_payload(task_input.task_family, best.reference_id),
                score_ppm=best.score_ppm,
                evidence=(f"linear:{self.linear_form.value}",),
                ranking=ranked,
            )
        return _CascadeResult(
            form=LocalExpertForm.ABSTAIN,
            reasons=(REASON_NO_DETERMINISTIC_MATCH,),
        )

    def predict(
        self,
        task_input: ResidualTaskInput,
        *,
        independent_validation: IndependentValidationReceipt | None = None,
    ) -> LocalExpertPrediction:
        if not isinstance(task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("predict requires ResidualTaskInput")
        feature_ops = len(self.feature_names) + 4
        cascade = self._cascade(task_input)
        return self._finalize(
            task_input,
            cascade,
            feature_identity=self._feature_identity(task_input),
            feature_ops=feature_ops,
            independent_validation=independent_validation,
        )

    def predict_batch(self, request: BatchedExpertRequest) -> tuple[LocalExpertPrediction, ...]:
        if not isinstance(request, BatchedExpertRequest):
            raise ResidualIntelligenceError("predict_batch requires BatchedExpertRequest")
        return tuple(
            self.predict(item, independent_validation=request.receipt_for(index))
            for index, item in enumerate(request.task_inputs)
        )

    def evaluate(self, cases: Sequence[ExpertEvaluationCase]) -> ExpertEvaluation:
        predictions = tuple(
            self.predict(case.task_input, independent_validation=case.independent_validation)
            for case in cases
        )
        return evaluate_local_predictions(
            cases, predictions, group_key=self.calibration_group.group_key
        )

    def fit(
        self,
        *,
        admission: TrainingCorpusAdmission,
        cases: Sequence[ExpertEvaluationCase],
        steps: int = 1,
        wall_seconds: int = 0,
        gpu_seconds: int = 0,
    ) -> LocalClassificationExpert:
        require_training_admission(admission)
        bounded_int(steps, "steps", minimum=1, maximum=MAX_LOCAL_STEPS)
        bounded_int(wall_seconds, "wall_seconds", minimum=0, maximum=MAX_LOCAL_WALL_SECONDS)
        gpu = bounded_int(gpu_seconds, "gpu_seconds", minimum=0, maximum=MAX_LOCAL_GPU_SECONDS)
        if gpu != 0:
            raise ResidualIntelligenceError(REASON_GPU_FORBIDDEN)
        if len(cases) > MAX_LOCAL_EXAMPLES:
            raise ResidualIntelligenceError(f"local fit exceeds {MAX_LOCAL_EXAMPLES} examples")
        if not cases:
            raise ResidualIntelligenceError("local fit requires at least one example")
        if self.checkpoint_count >= MAX_LOCAL_CHECKPOINTS:
            raise ResidualIntelligenceError(
                f"local fit exceeds {MAX_LOCAL_CHECKPOINTS} checkpoints"
            )
        labels = self.class_labels
        coefficients, intercepts, fitted_labels = fit_linear_coefficients(
            cases,
            feature_names=self.feature_names,
            class_labels=labels,
        )
        lookup = distill_exact_lookup(cases, feature_names=self.feature_names)
        return LocalClassificationExpert(
            schema=self.schema,
            task_family=self.task_family,
            expert_class=self.expert_class,
            calibration_group=self.calibration_group,
            feature_names=self.feature_names,
            lookup=lookup or self.lookup,
            rules=self.rules,
            class_labels=fitted_labels,
            linear_form=self.linear_form,
            coefficients=coefficients,
            intercepts=intercepts,
            linear_threshold_ppm=self.linear_threshold_ppm,
            selective_policy=self.selective_policy,
            ood_reference=self.ood_reference,
            ood_boundary=self.ood_boundary,
            policy_admits_ood=self.policy_admits_ood,
            ood_schema=self.ood_schema,
            ood_operation=self.ood_operation,
            ood_effects=self.ood_effects,
            ood_capabilities=self.ood_capabilities,
            ood_context_fields=self.ood_context_fields,
            fitted=True,
            admission_id=admission.admission_id,
            checkpoint_count=self.checkpoint_count + 1,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "expert_class": self.expert_class.value,
            "calibration_group": self.calibration_group.to_dict(),
            "feature_names": list(self.feature_names),
            "lookup": [item.to_dict() for item in self.lookup],
            "rules": [item.to_dict() for item in self.rules],
            "class_labels": list(self.class_labels),
            "linear_form": self.linear_form.value,
            "coefficients": [list(row) for row in self.coefficients],
            "intercepts": list(self.intercepts),
            "linear_threshold_ppm": self.linear_threshold_ppm,
            "selective_policy": (
                None if self.selective_policy is None else self.selective_policy.to_dict()
            ),
            "ood_reference": None if self.ood_reference is None else self.ood_reference.to_dict(),
            "ood_boundary": None if self.ood_boundary is None else self.ood_boundary.to_dict(),
            "policy_admits_ood": self.policy_admits_ood,
            "ood_schema": self.ood_schema,
            "ood_operation": self.ood_operation,
            "ood_effects": list(self.ood_effects),
            "ood_capabilities": list(self.ood_capabilities),
            "ood_context_fields": list(self.ood_context_fields),
            "fitted": self.fitted,
            "admission_id": self.admission_id,
            "checkpoint_count": self.checkpoint_count,
        }
        if include_id:
            result["expert_version"] = self.expert_version
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LocalClassificationExpert:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {
                "expert_version",
                "lookup",
                "rules",
                "class_labels",
                "coefficients",
                "intercepts",
                "linear_threshold_ppm",
                "selective_policy",
                "ood_reference",
                "ood_boundary",
                "policy_admits_ood",
                "ood_schema",
                "ood_operation",
                "ood_effects",
                "ood_capabilities",
                "ood_context_fields",
                "fitted",
                "admission_id",
                "checkpoint_count",
            },
            noun="local classification expert",
        )
        policy_payload = payload.get("selective_policy")
        reference_payload = payload.get("ood_reference")
        boundary_payload = payload.get("ood_boundary")
        group_payload = payload.get("calibration_group")
        if not isinstance(group_payload, Mapping):
            raise ResidualIntelligenceError("calibration_group must be an object")
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            expert_class=parse_expert_class(str(payload.get("expert_class") or "")),
            calibration_group=CalibrationGroup.from_dict(group_payload),
            feature_names=tuple(payload.get("feature_names") or ()),
            lookup=tuple(payload.get("lookup") or ()),
            rules=tuple(payload.get("rules") or ()),
            class_labels=tuple(payload.get("class_labels") or ()),
            linear_form=LinearForm(str(payload.get("linear_form") or LinearForm.LOGISTIC.value)),
            coefficients=tuple(tuple(row) for row in (payload.get("coefficients") or ())),
            intercepts=tuple(payload.get("intercepts") or ()),
            linear_threshold_ppm=payload.get("linear_threshold_ppm", 500_000),
            selective_policy=(
                None
                if policy_payload in (None, {})
                else SelectivePredictionPolicy.from_dict(policy_payload)
            ),
            ood_reference=(
                None
                if reference_payload in (None, {})
                else ReferenceDistribution.from_dict(reference_payload)
            ),
            ood_boundary=(
                None
                if boundary_payload in (None, {})
                else BoundaryContract.from_dict(boundary_payload)
            ),
            policy_admits_ood=payload.get("policy_admits_ood", False),
            ood_schema=str(payload.get("ood_schema") or ""),
            ood_operation=str(payload.get("ood_operation") or ""),
            ood_effects=tuple(payload.get("ood_effects") or ()),
            ood_capabilities=tuple(payload.get("ood_capabilities") or ()),
            ood_context_fields=tuple(payload.get("ood_context_fields") or ()),
            fitted=payload.get("fitted", False),
            admission_id=str(payload.get("admission_id") or ""),
            checkpoint_count=payload.get("checkpoint_count", 0),
        )
        claimed = str(payload.get("expert_version") or "")
        if claimed and claimed != result.expert_version:
            raise ResidualIntelligenceError("local classification expert version mismatch")
        return result


@dataclass(frozen=True)
class LocalRankingExpert(_LocalExpertMixin):
    """Exact/deterministic/small-ranking candidate for one ranking family."""

    task_family: ResidualTaskFamily
    expert_class: ExpertClass
    calibration_group: CalibrationGroup
    feature_names: tuple[str, ...]
    lookup: tuple[ExactLookupEntry, ...] = ()
    ranking_weights: tuple[tuple[str, int], ...] = ()
    small_ranker: SmallRanker = None  # type: ignore[assignment]
    selective_policy: SelectivePredictionPolicy | None = None
    ood_reference: ReferenceDistribution | None = None
    ood_boundary: BoundaryContract | None = None
    policy_admits_ood: bool = False
    ood_schema: str = ""
    ood_operation: str = ""
    ood_effects: tuple[str, ...] = ()
    ood_capabilities: tuple[str, ...] = ()
    ood_context_fields: tuple[str, ...] = ()
    fitted: bool = False
    admission_id: str = ""
    checkpoint_count: int = 0
    schema: str = LOCAL_RANKING_EXPERT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "expert_version",
            "task_family",
            "expert_class",
            "calibration_group",
            "feature_names",
            "lookup",
            "ranking_weights",
            "small_ranker",
            "selective_policy",
            "ood_reference",
            "ood_boundary",
            "policy_admits_ood",
            "ood_schema",
            "ood_operation",
            "ood_effects",
            "ood_capabilities",
            "ood_context_fields",
            "fitted",
            "admission_id",
            "checkpoint_count",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != LOCAL_RANKING_EXPERT_SCHEMA:
            raise ResidualIntelligenceError("unsupported local ranking expert schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "expert_class", parse_expert_class(self.expert_class))
        family_spec, spec = _bind_local_expert(
            task_family=self.task_family,
            expert_class=self.expert_class,
            calibration_group=self.calibration_group,
            expected_kind=SEMANTIC_KIND_RANKING,
        )
        if spec.expert_class is not self.expert_class:
            raise ResidualIntelligenceError("ranking expert class mismatch")
        if family_spec.semantic_kind != SEMANTIC_KIND_RANKING:
            raise ResidualIntelligenceError(REASON_UNSUPPORTED_KIND)
        object.__setattr__(self, "feature_names", _feature_names(self.feature_names))
        object.__setattr__(self, "lookup", _lookup_entries(self.lookup))
        object.__setattr__(self, "ranking_weights", _weight_pairs(self.ranking_weights))
        ranker = self.small_ranker if self.small_ranker is not None else SmallRanker()
        if not isinstance(ranker, SmallRanker):
            raise ResidualIntelligenceError("small_ranker must be SmallRanker")
        object.__setattr__(self, "small_ranker", ranker)
        if self.selective_policy is not None and not isinstance(
            self.selective_policy, SelectivePredictionPolicy
        ):
            raise ResidualIntelligenceError("selective_policy must be SelectivePredictionPolicy")
        if self.ood_reference is not None and not isinstance(
            self.ood_reference, ReferenceDistribution
        ):
            raise ResidualIntelligenceError("ood_reference must be ReferenceDistribution")
        if self.ood_boundary is not None and not isinstance(self.ood_boundary, BoundaryContract):
            raise ResidualIntelligenceError("ood_boundary must be BoundaryContract")
        object.__setattr__(
            self, "policy_admits_ood", _require_bool(self.policy_admits_ood, "policy_admits_ood")
        )
        object.__setattr__(
            self,
            "ood_schema",
            "" if self.ood_schema in (None, "") else required_text(self.ood_schema, "ood_schema"),
        )
        object.__setattr__(
            self,
            "ood_operation",
            ""
            if self.ood_operation in (None, "")
            else required_text(self.ood_operation, "ood_operation"),
        )
        object.__setattr__(self, "ood_effects", text_tuple(self.ood_effects, "ood_effects"))
        object.__setattr__(
            self, "ood_capabilities", text_tuple(self.ood_capabilities, "ood_capabilities")
        )
        object.__setattr__(
            self,
            "ood_context_fields",
            text_tuple(self.ood_context_fields, "ood_context_fields"),
        )
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
                maximum=MAX_LOCAL_CHECKPOINTS,
            ),
        )
        if self.fitted and not self.admission_id:
            raise ResidualIntelligenceError("fitted local expert requires an admission_id")

    @property
    def expert_version(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def ranking_weight_map(self) -> dict[str, int]:
        return {key: value for key, value in self.ranking_weights}

    def _ranking_result(
        self,
        form: LocalExpertForm,
        ranking: tuple[RankingItem, ...],
        *,
        reason: str,
        evidence: tuple[str, ...],
    ) -> _CascadeResult:
        if not ranking:
            return _CascadeResult(form=LocalExpertForm.ABSTAIN, reasons=(REASON_EMPTY_RANKING,))
        if _has_score_tie(ranking) and form is LocalExpertForm.DETERMINISTIC_RANKING:
            return _CascadeResult(
                form=LocalExpertForm.ABSTAIN,
                reasons=(REASON_RANKING_TIE,),
                ranking=ranking,
            )
        if _has_score_tie(ranking) and form is LocalExpertForm.SMALL_RANKER:
            return _CascadeResult(
                form=LocalExpertForm.ABSTAIN,
                reasons=(REASON_RANKING_TIE,),
                ranking=ranking,
            )
        return _CascadeResult(
            form=form,
            reasons=(reason,),
            output_class=self.task_family.value,
            payload=ranking_payload(self.task_family, ranking),
            score_ppm=ranking[0].score_ppm,
            evidence=evidence,
            ranking=ranking,
        )

    def _cascade(self, task_input: ResidualTaskInput) -> _CascadeResult:
        rejected = _validate_family_and_risk(task_input, self.family_spec, self.expert_spec)
        if rejected:
            return _CascadeResult(form=LocalExpertForm.REJECT_INPUT, reasons=rejected)
        features = dict(task_input.compact_features)
        identity = self._feature_identity(task_input)
        if form_permitted(LocalExpertForm.EXACT_LOOKUP, self.expert_class):
            hit = self.lookup_by_identity().get(identity)
            if hit is not None:
                ranking = ()
                raw_ids = hit.structured_payload.get("ranked_reference_ids") or hit.structured_payload.get(
                    "test_ids"
                ) or hit.structured_payload.get("proof_ids")
                raw_scores = hit.structured_payload.get("scores_ppm")
                if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
                    scores = list(raw_scores or [hit.score_ppm] * len(raw_ids))
                    ranking = _ranking_tuple(
                        tuple(
                            RankingItem(reference_id=str(item), score_ppm=int(score))
                            for item, score in zip(raw_ids, scores)
                        )
                    )
                return _CascadeResult(
                    form=LocalExpertForm.EXACT_LOOKUP,
                    reasons=(REASON_EXACT_LOOKUP,),
                    output_class=hit.output_class,
                    payload=hit.structured_payload,
                    score_ppm=hit.score_ppm,
                    evidence=hit.evidence_references,
                    ranking=ranking,
                )
        if form_permitted(LocalExpertForm.DETERMINISTIC_RANKING, self.expert_class):
            ranking = _rank_candidates(features, self.ranking_weight_map())
            if ranking:
                result = self._ranking_result(
                    LocalExpertForm.DETERMINISTIC_RANKING,
                    ranking,
                    reason=REASON_DETERMINISTIC_RANKING,
                    evidence=("ranking:stable-sort",),
                )
                if result.form is not LocalExpertForm.ABSTAIN:
                    return result
                if not form_permitted(LocalExpertForm.SMALL_RANKER, self.expert_class):
                    if expert_class_rank(self.expert_class) >= expert_class_rank(ExpertClass.C):
                        return _CascadeResult(
                            form=LocalExpertForm.DETERMINISTIC_RANKING,
                            reasons=(REASON_DETERMINISTIC_RANKING,),
                            output_class=self.task_family.value,
                            payload=ranking_payload(self.task_family, ranking),
                            score_ppm=ranking[0].score_ppm,
                            evidence=("ranking:stable-sort",),
                            ranking=ranking,
                        )
                    return result
        else:
            ranking = ()
        if form_permitted(LocalExpertForm.SMALL_RANKER, self.expert_class):
            if not self.small_ranker.available:
                return _CascadeResult(
                    form=LocalExpertForm.ABSTAIN,
                    reasons=(REASON_SMALL_RANKER_UNAVAILABLE, REASON_NO_DETERMINISTIC_MATCH),
                )
            ranked = _rank_candidates(
                features,
                self.ranking_weight_map(),
                scale=max(1, self.small_ranker.signal_scale),
                extra_weights=self.small_ranker.weight_map(),
            )
            if self.small_ranker.intercept_ppm and ranked:
                ranked = _ranking_tuple(
                    tuple(
                        RankingItem(
                            reference_id=item.reference_id,
                            score_ppm=_clamp_ppm(item.score_ppm + self.small_ranker.intercept_ppm),
                        )
                        for item in ranked
                    )
                )
            return self._ranking_result(
                LocalExpertForm.SMALL_RANKER,
                ranked,
                reason=REASON_SMALL_RANKER,
                evidence=("ranking:small-ranker",),
            )
        if ranking:
            return _CascadeResult(form=LocalExpertForm.ABSTAIN, reasons=(REASON_RANKING_TIE,))
        return _CascadeResult(
            form=LocalExpertForm.ABSTAIN,
            reasons=(REASON_NO_DETERMINISTIC_MATCH,),
        )

    def predict(
        self,
        task_input: ResidualTaskInput,
        *,
        independent_validation: IndependentValidationReceipt | None = None,
    ) -> LocalExpertPrediction:
        if not isinstance(task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("predict requires ResidualTaskInput")
        feature_ops = len(self.feature_names) + 6
        cascade = self._cascade(task_input)
        return self._finalize(
            task_input,
            cascade,
            feature_identity=self._feature_identity(task_input),
            feature_ops=feature_ops,
            independent_validation=independent_validation,
        )

    def predict_batch(self, request: BatchedExpertRequest) -> tuple[LocalExpertPrediction, ...]:
        if not isinstance(request, BatchedExpertRequest):
            raise ResidualIntelligenceError("predict_batch requires BatchedExpertRequest")
        return tuple(
            self.predict(item, independent_validation=request.receipt_for(index))
            for index, item in enumerate(request.task_inputs)
        )

    def evaluate(self, cases: Sequence[ExpertEvaluationCase]) -> ExpertEvaluation:
        predictions = tuple(
            self.predict(case.task_input, independent_validation=case.independent_validation)
            for case in cases
        )
        return evaluate_local_predictions(
            cases, predictions, group_key=self.calibration_group.group_key
        )

    def fit(
        self,
        *,
        admission: TrainingCorpusAdmission,
        cases: Sequence[ExpertEvaluationCase],
        steps: int = 1,
        wall_seconds: int = 0,
        gpu_seconds: int = 0,
    ) -> LocalRankingExpert:
        require_training_admission(admission)
        bounded_int(steps, "steps", minimum=1, maximum=MAX_LOCAL_STEPS)
        bounded_int(wall_seconds, "wall_seconds", minimum=0, maximum=MAX_LOCAL_WALL_SECONDS)
        gpu = bounded_int(gpu_seconds, "gpu_seconds", minimum=0, maximum=MAX_LOCAL_GPU_SECONDS)
        if gpu != 0:
            raise ResidualIntelligenceError(REASON_GPU_FORBIDDEN)
        if len(cases) > MAX_LOCAL_EXAMPLES:
            raise ResidualIntelligenceError(f"local fit exceeds {MAX_LOCAL_EXAMPLES} examples")
        if not cases:
            raise ResidualIntelligenceError("local fit requires at least one example")
        if self.checkpoint_count >= MAX_LOCAL_CHECKPOINTS:
            raise ResidualIntelligenceError(
                f"local fit exceeds {MAX_LOCAL_CHECKPOINTS} checkpoints"
            )
        ranker = fit_small_ranker(cases, admission_id=admission.admission_id)
        lookup = distill_exact_lookup(cases, feature_names=self.feature_names)
        return LocalRankingExpert(
            schema=self.schema,
            task_family=self.task_family,
            expert_class=self.expert_class,
            calibration_group=self.calibration_group,
            feature_names=self.feature_names,
            lookup=lookup or self.lookup,
            ranking_weights=self.ranking_weights,
            small_ranker=ranker,
            selective_policy=self.selective_policy,
            ood_reference=self.ood_reference,
            ood_boundary=self.ood_boundary,
            policy_admits_ood=self.policy_admits_ood,
            ood_schema=self.ood_schema,
            ood_operation=self.ood_operation,
            ood_effects=self.ood_effects,
            ood_capabilities=self.ood_capabilities,
            ood_context_fields=self.ood_context_fields,
            fitted=True,
            admission_id=admission.admission_id,
            checkpoint_count=self.checkpoint_count + 1,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "expert_class": self.expert_class.value,
            "calibration_group": self.calibration_group.to_dict(),
            "feature_names": list(self.feature_names),
            "lookup": [item.to_dict() for item in self.lookup],
            "ranking_weights": [[key, value] for key, value in self.ranking_weights],
            "small_ranker": self.small_ranker.to_dict(),
            "selective_policy": (
                None if self.selective_policy is None else self.selective_policy.to_dict()
            ),
            "ood_reference": None if self.ood_reference is None else self.ood_reference.to_dict(),
            "ood_boundary": None if self.ood_boundary is None else self.ood_boundary.to_dict(),
            "policy_admits_ood": self.policy_admits_ood,
            "ood_schema": self.ood_schema,
            "ood_operation": self.ood_operation,
            "ood_effects": list(self.ood_effects),
            "ood_capabilities": list(self.ood_capabilities),
            "ood_context_fields": list(self.ood_context_fields),
            "fitted": self.fitted,
            "admission_id": self.admission_id,
            "checkpoint_count": self.checkpoint_count,
        }
        if include_id:
            result["expert_version"] = self.expert_version
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LocalRankingExpert:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {
                "expert_version",
                "lookup",
                "ranking_weights",
                "small_ranker",
                "selective_policy",
                "ood_reference",
                "ood_boundary",
                "policy_admits_ood",
                "ood_schema",
                "ood_operation",
                "ood_effects",
                "ood_capabilities",
                "ood_context_fields",
                "fitted",
                "admission_id",
                "checkpoint_count",
            },
            noun="local ranking expert",
        )
        policy_payload = payload.get("selective_policy")
        reference_payload = payload.get("ood_reference")
        boundary_payload = payload.get("ood_boundary")
        ranker_payload = payload.get("small_ranker")
        group_payload = payload.get("calibration_group")
        if not isinstance(group_payload, Mapping):
            raise ResidualIntelligenceError("calibration_group must be an object")
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            expert_class=parse_expert_class(str(payload.get("expert_class") or "")),
            calibration_group=CalibrationGroup.from_dict(group_payload),
            feature_names=tuple(payload.get("feature_names") or ()),
            lookup=tuple(payload.get("lookup") or ()),
            ranking_weights=tuple(payload.get("ranking_weights") or ()),
            small_ranker=(
                SmallRanker()
                if ranker_payload in (None, {})
                else SmallRanker.from_dict(ranker_payload)
            ),
            selective_policy=(
                None
                if policy_payload in (None, {})
                else SelectivePredictionPolicy.from_dict(policy_payload)
            ),
            ood_reference=(
                None
                if reference_payload in (None, {})
                else ReferenceDistribution.from_dict(reference_payload)
            ),
            ood_boundary=(
                None
                if boundary_payload in (None, {})
                else BoundaryContract.from_dict(boundary_payload)
            ),
            policy_admits_ood=payload.get("policy_admits_ood", False),
            ood_schema=str(payload.get("ood_schema") or ""),
            ood_operation=str(payload.get("ood_operation") or ""),
            ood_effects=tuple(payload.get("ood_effects") or ()),
            ood_capabilities=tuple(payload.get("ood_capabilities") or ()),
            ood_context_fields=tuple(payload.get("ood_context_fields") or ()),
            fitted=payload.get("fitted", False),
            admission_id=str(payload.get("admission_id") or ""),
            checkpoint_count=payload.get("checkpoint_count", 0),
        )
        claimed = str(payload.get("expert_version") or "")
        if claimed and claimed != result.expert_version:
            raise ResidualIntelligenceError("local ranking expert version mismatch")
        return result


def distill_exact_lookup(
    cases: Sequence[ExpertEvaluationCase],
    *,
    feature_names: Sequence[str],
) -> tuple[ExactLookupEntry, ...]:
    grouped: dict[str, list[ExpertEvaluationCase]] = {}
    for case in cases:
        if case.expected_output_class == ABSTAIN_OUTPUT_CLASS:
            continue
        if case.partition not in {SplitPartition.TRAIN, SplitPartition.DEVELOPMENT}:
            continue
        identity = local_feature_identity(case.task_input.compact_features, feature_names)
        grouped.setdefault(identity, []).append(case)
    entries: list[ExactLookupEntry] = []
    for identity, rows in sorted(grouped.items()):
        labels = {row.expected_output_class for row in rows}
        if len(labels) != 1:
            continue
        payloads = [dict(row.expected_payload) for row in rows if row.expected_payload]
        payload = payloads[0] if payloads else classification_payload(
            rows[0].task_input.task_family, rows[0].expected_output_class
        )
        if any(item != payload for item in payloads[1:]):
            continue
        entries.append(
            ExactLookupEntry(
                feature_identity=identity,
                output_class=rows[0].expected_output_class,
                structured_payload=payload,
                score_ppm=MAX_SCORE_PPM,
                evidence_references=tuple(
                    item.example_identity for item in rows if item.example_identity
                )[:16],
            )
        )
        if len(entries) >= MAX_LOOKUP_ENTRIES:
            break
    return tuple(entries)


def _observed_linear_label(
    case: ExpertEvaluationCase, class_labels: Sequence[str]
) -> str:
    payload = case.expected_payload or {}
    for key in (
        "failure_class",
        "label",
        "decision",
        "claim_class",
        "conflict_class",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    observed = case.expected_output_class
    if observed == ABSTAIN_OUTPUT_CLASS:
        return observed
    if class_labels and observed not in class_labels:
        if observed == case.task_input.task_family.value and len(class_labels) == 1:
            return class_labels[0]
    return observed


def fit_linear_coefficients(
    cases: Sequence[ExpertEvaluationCase],
    *,
    feature_names: Sequence[str],
    class_labels: Sequence[str] = (),
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[str, ...]]:
    ordered = tuple(
        sorted(
            cases,
            key=lambda item: (item.task_input.input_id, item.expected_output_class),
        )
    )
    rows: list[tuple[tuple[int, ...], str]] = []
    for case in ordered:
        if case.partition not in {SplitPartition.TRAIN, SplitPartition.DEVELOPMENT}:
            continue
        vector = extract_local_linear_vector(case.task_input.compact_features, feature_names)
        if vector is None:
            continue
        rows.append((vector, _observed_linear_label(case, class_labels)))
    labels = tuple(class_labels) or tuple(
        sorted({label for _vector, label in rows if label != ABSTAIN_OUTPUT_CLASS})
    )
    if not labels:
        raise ResidualIntelligenceError("linear fit requires a non-abstain class label")
    n_features = len(feature_names)
    coefficients: list[tuple[int, ...]] = []
    intercepts: list[int] = []
    for label in labels:
        positives = [vector for vector, observed in rows if observed == label]
        negatives = [vector for vector, observed in rows if observed != label]
        row: list[int] = []
        for index in range(n_features):
            pos_mean = (
                0
                if not positives
                else (sum(item[index] for item in positives) * LOGIT_SCALE) // len(positives)
            )
            neg_mean = (
                0
                if not negatives
                else (sum(item[index] for item in negatives) * LOGIT_SCALE) // len(negatives)
            )
            delta = pos_mean - neg_mean
            if delta > MAX_COEFFICIENT:
                delta = MAX_COEFFICIENT
            elif delta < -MAX_COEFFICIENT:
                delta = -MAX_COEFFICIENT
            row.append(delta)
        prior_num = len(positives)
        prior_den = len(rows) or 1
        intercept = 0
        if 0 < prior_num < prior_den:
            intercept = ((2 * prior_num - prior_den) * LOGIT_SCALE) // prior_den
        intercepts.append(max(-MAX_COEFFICIENT, min(MAX_COEFFICIENT, intercept)))
        coefficients.append(tuple(row))
    return tuple(coefficients), tuple(intercepts), labels


def fit_small_ranker(
    cases: Sequence[ExpertEvaluationCase],
    *,
    admission_id: str,
) -> SmallRanker:
    weights: dict[str, int] = {}
    for case in cases:
        if case.partition not in {SplitPartition.TRAIN, SplitPartition.DEVELOPMENT}:
            continue
        ids = case.expected_payload.get("ranked_reference_ids") or case.expected_payload.get(
            "test_ids"
        ) or case.expected_payload.get("proof_ids")
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
            continue
        total = len(ids)
        for index, reference in enumerate(ids):
            key = _reject_private_name(str(reference), "ranking candidate")
            gain = _clamp_ppm((total - index) * LOGIT_SCALE)
            weights[key] = _clamp_ppm(weights.get(key, 0) + gain)
    pairs = tuple(sorted(weights.items(), key=lambda item: item[0]))
    return SmallRanker(
        candidate_weights=pairs,
        signal_scale=1,
        intercept_ppm=0,
        fitted=True,
        admission_id=admission_id,
        checkpoint_count=1,
    )


def admit_local_expert_class(
    task_family: ResidualTaskFamily | str,
    requested: ExpertClass | str,
    *,
    risk: RiskClass | str,
    quality_delta_ppm: int = 0,
    routing_changing: bool = False,
    evidence_current: bool = False,
    compared_class: ExpertClass | str | None = None,
    admission: TrainingCorpusAdmission | None = None,
) -> ResidualExpertSpec:
    family = ResidualTaskFamily(task_family)
    spec = family_spec_for(family)
    if spec.semantic_kind not in {SEMANTIC_KIND_CLASSIFICATION, SEMANTIC_KIND_RANKING}:
        raise ResidualIntelligenceError(REASON_UNSUPPORTED_KIND)
    parsed_risk = RiskClass(risk)
    if parsed_risk not in LOW_MEDIUM_RISKS and parsed_risk not in PROPOSAL_RISKS:
        raise ResidualIntelligenceError(REASON_LOW_MEDIUM_RISK)
    return admit_expert_class(
        family,
        requested,
        risk=parsed_risk,
        quality_delta_ppm=quality_delta_ppm,
        routing_changing=routing_changing,
        evidence_current=evidence_current,
        compared_class=compared_class,
        admission=admission,
    )


__all__ = (
    "BATCHED_EXPERT_REQUEST_SCHEMA",
    "EXPERT_EVALUATION_SCHEMA",
    "LOW_MEDIUM_RISKS",
    "MAX_BATCH_ITEMS",
    "MAX_LOCAL_CHECKPOINTS",
    "MAX_LOCAL_EXAMPLES",
    "MAX_LOCAL_GPU_SECONDS",
    "MAX_LOCAL_STEPS",
    "MAX_LOCAL_WALL_SECONDS",
    "MIN_ROUTING_CHANGING_DELTA_PPM",
    "REASON_TRAINING_UNAVAILABLE",
    "BatchedExpertRequest",
    "ExpertEvaluation",
    "ExpertEvaluationCase",
    "IndependentValidationReceipt",
    "LocalClassificationExpert",
    "LocalExpertCostReceipt",
    "LocalExpertForm",
    "LocalExpertPrediction",
    "LocalRankingExpert",
    "SmallRanker",
    "admit_local_expert_class",
    "classification_payload",
    "distill_exact_lookup",
    "evaluate_local_predictions",
    "fit_linear_coefficients",
    "fit_small_ranker",
    "form_permitted",
    "local_feature_identity",
    "ranking_payload",
    "require_training_admission",
    "structured_output_valid",
)
