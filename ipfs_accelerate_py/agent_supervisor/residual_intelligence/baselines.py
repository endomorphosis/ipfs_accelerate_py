"""Deterministic and bounded linear residual baselines.

Exact lookup, verified-procedure compatibility, declarative rules, and
stable rankings precede any linear or logistic score.  Learned and remote
routes are out of scope: a baseline never invokes a provider, even when a
procedure precondition fails.  Evaluation receipts keep complete
denominators rather than rates alone.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

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
from .residual_ir import (
    MAX_SCORE_PPM,
    ResidualIntelligenceIR,
    ResidualTaskInput,
    ResidualTaskOutput,
)
from .rights import TrainingCorpusAdmission

BASELINE_COST_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-baseline-cost-receipt@1"
)
BASELINE_PREDICTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-baseline-prediction@1"
)
BASELINE_EVALUATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-baseline-evaluation@1"
)
EXACT_LOOKUP_ENTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-baseline-exact-lookup@1"
)
RULE_PREDICATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-baseline-rule-predicate@1"
)
DECLARATIVE_RULE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-baseline-declarative-rule@1"
)
PROCEDURE_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-baseline-procedure-binding@1"
)
RANKING_ITEM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-baseline-ranking-item@1"
)
DETERMINISTIC_EXPERT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-deterministic-expert@1"
)
LINEAR_EXPERT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-linear-expert@1"
)
STABLE_FEATURE_VECTOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-baseline-stable-features@1"
)

MAX_LOOKUP_ENTRIES: Final = 1_024
MAX_RULES: Final = 256
MAX_PREDICATES: Final = 16
MAX_FEATURES: Final = 64
MAX_FEATURE_VALUE: Final = 1_000_000
MAX_COEFFICIENT: Final = 1_000_000
MAX_LINEAR_EXAMPLES: Final = 10_000
MAX_RANKING_ITEMS: Final = 256
MAX_FIT_CPU_SECONDS: Final = 1_800
MAX_FIT_CHECKPOINTS: Final = 1
LOGIT_SCALE: Final = 1_000
LOGIT_CLAMP: Final = 20_000
ABSTAIN_OUTPUT_CLASS: Final = "ABSTAIN"
PROPOSAL_RISKS: Final[frozenset[RiskClass]] = frozenset({RiskClass.R4, RiskClass.R5})

FEATURE_CRITICAL_BOUNDARY: Final = "critical_boundary"
FEATURE_INPUT_VALID: Final = "input_valid"
FEATURE_PROCEDURE_ROOT: Final = "procedure_root"
FEATURE_PROCEDURE_ANSWER: Final = "procedure_answer_available"
FEATURE_PROCEDURE_PRECONDITIONS: Final = "procedure_preconditions_satisfied"
FEATURE_RANKING_CANDIDATES: Final = "ranking_candidates"
FEATURE_RANKING_SIGNALS: Final = "ranking_signals"

REASON_REJECT_INPUT: Final = "reject_input"
REASON_FAMILY_MISMATCH: Final = "task_family_mismatch"
REASON_CRITICAL_BOUNDARY: Final = "critical_boundary_abstention"
REASON_PROCEDURE_PRECONDITION: Final = "procedure_precondition_failure"
REASON_PROCEDURE_UNBOUND: Final = "procedure_binding_missing"
REASON_NO_DETERMINISTIC_MATCH: Final = "no_deterministic_match"
REASON_MISSING_STABLE_FEATURE: Final = "missing_stable_feature"
REASON_LINEAR_UNAVAILABLE: Final = "linear_coefficients_unavailable"
REASON_LINEAR_NO_SIGNAL: Final = "linear_no_signal"
REASON_OUTPUT_NOT_ALLOWED: Final = "output_class_not_allowed"
REASON_VALIDATION_REQUIRED: Final = "VALIDATION_REQUIRED"
REASON_R4_R5_PROPOSAL: Final = "r4_r5_proposal_tier"
REASON_EXACT_LOOKUP: Final = "exact_lookup"
REASON_VERIFIED_PROCEDURE: Final = "verified_procedure"
REASON_DETERMINISTIC_RULE: Final = "deterministic_rule"
REASON_DETERMINISTIC_RANKING: Final = "deterministic_ranking"
REASON_LINEAR_LOGISTIC: Final = "linear_logistic"
REASON_TRAINING_UNAVAILABLE: Final = "training_unavailable"

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


class BaselineRoute(str, Enum):
    """Closed producer used for one baseline prediction."""

    EXACT_LOOKUP = "exact_lookup"
    VERIFIED_PROCEDURE = "verified_procedure"
    DETERMINISTIC_RULE = "deterministic_rule"
    DETERMINISTIC_RANKING = "deterministic_ranking"
    LINEAR_LOGISTIC = "linear_logistic"
    ABSTAIN = "abstain"
    REJECT_INPUT = "reject_input"


class LinearForm(str, Enum):
    LINEAR = "linear"
    LOGISTIC = "logistic"


class RulePredicateKind(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    PRESENT = "present"
    ABSENT = "absent"
    BOOL_TRUE = "bool_true"
    BOOL_FALSE = "bool_false"
    INT_EQUALS = "int_equals"
    INT_AT_LEAST = "int_at_least"
    INT_AT_MOST = "int_at_most"


BASELINE_CASCADE_ORDER: Final[tuple[BaselineRoute, ...]] = (
    BaselineRoute.EXACT_LOOKUP,
    BaselineRoute.VERIFIED_PROCEDURE,
    BaselineRoute.DETERMINISTIC_RULE,
    BaselineRoute.DETERMINISTIC_RANKING,
    BaselineRoute.LINEAR_LOGISTIC,
)

_PRODUCER_ROUTES: Final[frozenset[BaselineRoute]] = frozenset(BASELINE_CASCADE_ORDER)
_MISSING: Final[object] = object()


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
        raise ResidualIntelligenceError(
            f"{name} cannot memorize or expose a private body"
        )
    return text


def _reject_private_mapping(value: Mapping[str, Any], *, noun: str) -> dict[str, Any]:
    payload = bounded_json_mapping(value, noun)
    reject_secret_material(payload, noun=noun)
    reject_candidate_authority(payload)

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                text = str(key)
                if _looks_like_private_body(text):
                    raise ResidualIntelligenceError(
                        f"{noun} exposes a private body at {path}{text}"
                    )
                visit(child, f"{path}{text}.")
        elif isinstance(item, str) and _looks_like_private_body(item):
            raise ResidualIntelligenceError(f"{noun} exposes a private body at {path}")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, f"{path}{index}.")

    visit(payload, "")
    return payload


def _feature_names(values: Any) -> tuple[str, ...]:
    names = text_tuple(values, "feature_names", max_items=MAX_FEATURES)
    return tuple(_reject_private_name(item, "feature_names item") for item in names)


def logistic_ppm(logit: int) -> int:
    """Bit-exact bounded logistic in parts-per-million.

    Uses the integer fast sigmoid ``(x + |x| + s) / (2 (|x| + s))`` with
    ``s = 1000`` milli-logits so coefficients remain reproducible without a
    floating-point runtime.
    """

    if type(logit) is not int:
        raise ResidualIntelligenceError("logistic logit must be an integer")
    clamped = max(-LOGIT_CLAMP, min(LOGIT_CLAMP, logit))
    if clamped >= LOGIT_CLAMP:
        return MAX_SCORE_PPM
    if clamped <= -LOGIT_CLAMP:
        return 0
    magnitude = abs(clamped)
    return ((clamped + magnitude + LOGIT_SCALE) * (MAX_SCORE_PPM // 2)) // (
        magnitude + LOGIT_SCALE
    )


def _clamp_ppm(value: int) -> int:
    if value < 0:
        return 0
    if value > MAX_SCORE_PPM:
        return MAX_SCORE_PPM
    return value


def _stable_scalar(value: Any, name: str) -> str | int | bool:
    if type(value) is bool:
        return value
    if type(value) is int:
        return bounded_int(value, name, minimum=-MAX_FEATURE_VALUE, maximum=MAX_FEATURE_VALUE)
    if isinstance(value, str):
        return _reject_private_name(value, name)
    raise ResidualIntelligenceError(f"{name} is not a stable scalar feature")


def extract_stable_features(
    compact_features: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, str | int | bool]:
    """Project declared features into a canonical, body-free vector."""

    if not isinstance(compact_features, Mapping):
        raise ResidualIntelligenceError("compact_features must be an object")
    _reject_private_mapping(dict(compact_features), noun="compact_features")
    extracted: dict[str, str | int | bool] = {}
    for name in feature_names:
        if name in compact_features:
            extracted[name] = _stable_scalar(compact_features[name], name)
    return extracted


def stable_feature_identity(
    compact_features: Mapping[str, Any],
    feature_names: Sequence[str],
) -> str:
    extracted = extract_stable_features(compact_features, feature_names)
    return canonical_id(
        {
            "schema": STABLE_FEATURE_VECTOR_SCHEMA,
            "feature_names": list(feature_names),
            "values": extracted,
        }
    )


def _numeric_feature(value: Any, name: str) -> int:
    if type(value) is bool:
        return 1 if value else 0
    return bounded_int(value, name, minimum=0, maximum=MAX_FEATURE_VALUE)


def extract_linear_vector(
    compact_features: Mapping[str, Any],
    feature_names: Sequence[str],
) -> tuple[int, ...] | None:
    if any(name not in compact_features for name in feature_names):
        return None
    try:
        return tuple(
            _numeric_feature(compact_features[name], name) for name in feature_names
        )
    except ResidualIntelligenceError:
        return None


@dataclass(frozen=True)
class RankingItem:
    """One stably ordered ranking candidate."""

    reference_id: str
    score_ppm: int
    schema: str = RANKING_ITEM_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema", "item_id", "reference_id", "score_ppm"}
    )

    def __post_init__(self) -> None:
        if self.schema != RANKING_ITEM_SCHEMA:
            raise ResidualIntelligenceError("unsupported ranking item schema")
        object.__setattr__(
            self, "reference_id", _reject_private_name(self.reference_id, "reference_id")
        )
        object.__setattr__(self, "score_ppm", _ppm(self.score_ppm, "score_ppm"))

    @property
    def item_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "reference_id": self.reference_id,
            "score_ppm": self.score_ppm,
        }
        if include_id:
            result["item_id"] = self.item_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RankingItem:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"item_id"},
            noun="ranking item",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            reference_id=str(payload.get("reference_id") or ""),
            score_ppm=payload.get("score_ppm"),
        )
        claimed = str(payload.get("item_id") or "")
        if claimed and claimed != result.item_id:
            raise ResidualIntelligenceError("ranking item identity mismatch")
        return result


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


@dataclass(frozen=True)
class BaselineCostReceipt:
    """Local-only cost record; remote/model fields are hard-zero."""

    route: BaselineRoute
    feature_ops: int
    avoided_remote_calls: int
    avoided_strong_calls: int
    model_calls: int = 0
    provider_invocations: int = 0
    remote_input_tokens: int = 0
    remote_output_tokens: int = 0
    cost_microunits: int = 0
    latency_ms: int = 0
    schema: str = BASELINE_COST_RECEIPT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "receipt_id",
            "route",
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
        if self.schema != BASELINE_COST_RECEIPT_SCHEMA:
            raise ResidualIntelligenceError("unsupported baseline cost receipt schema")
        object.__setattr__(self, "route", BaselineRoute(self.route))
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
                "baseline cost receipts cannot record a model or provider call"
            )
        if self.remote_input_tokens != 0 or self.remote_output_tokens != 0:
            raise ResidualIntelligenceError(
                "baseline cost receipts cannot record remote tokens"
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
            "route": self.route.value,
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
    def from_dict(cls, payload: Mapping[str, Any]) -> BaselineCostReceipt:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"receipt_id"},
            noun="baseline cost receipt",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            route=BaselineRoute(str(payload.get("route") or "")),
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
            raise ResidualIntelligenceError("baseline cost receipt identity mismatch")
        return result


def _local_cost(route: BaselineRoute, *, feature_ops: int) -> BaselineCostReceipt:
    return BaselineCostReceipt(
        route=route,
        feature_ops=feature_ops,
        avoided_remote_calls=1,
        avoided_strong_calls=1,
    )


@dataclass(frozen=True)
class ExactLookupEntry:
    """Exact cache row keyed by a stable feature identity, never a private body."""

    feature_identity: str
    output_class: str
    structured_payload: Mapping[str, Any]
    score_ppm: int
    evidence_references: tuple[str, ...] = ()
    schema: str = EXACT_LOOKUP_ENTRY_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "entry_id",
            "feature_identity",
            "output_class",
            "structured_payload",
            "score_ppm",
            "evidence_references",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != EXACT_LOOKUP_ENTRY_SCHEMA:
            raise ResidualIntelligenceError("unsupported exact lookup schema")
        object.__setattr__(
            self,
            "feature_identity",
            required_text(self.feature_identity, "feature_identity"),
        )
        object.__setattr__(
            self,
            "output_class",
            _reject_private_name(self.output_class, "output_class"),
        )
        object.__setattr__(
            self,
            "structured_payload",
            _reject_private_mapping(self.structured_payload, noun="structured_payload"),
        )
        object.__setattr__(self, "score_ppm", _ppm(self.score_ppm, "score_ppm"))
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references", max_items=256),
        )

    @property
    def entry_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "feature_identity": self.feature_identity,
            "output_class": self.output_class,
            "structured_payload": dict(self.structured_payload),
            "score_ppm": self.score_ppm,
            "evidence_references": list(self.evidence_references),
        }
        if include_id:
            result["entry_id"] = self.entry_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExactLookupEntry:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"entry_id"},
            noun="exact lookup entry",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            feature_identity=str(payload.get("feature_identity") or ""),
            output_class=str(payload.get("output_class") or ""),
            structured_payload=payload.get("structured_payload") or {},
            score_ppm=payload.get("score_ppm"),
            evidence_references=tuple(payload.get("evidence_references") or ()),
        )
        claimed = str(payload.get("entry_id") or "")
        if claimed and claimed != result.entry_id:
            raise ResidualIntelligenceError("exact lookup identity mismatch")
        return result


@dataclass(frozen=True)
class RulePredicate:
    """One closed predicate over a compact feature; never over a private body."""

    feature: str
    kind: RulePredicateKind
    value: Any = ""
    schema: str = RULE_PREDICATE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"schema", "predicate_id", "feature", "kind", "value"}
    )

    def __post_init__(self) -> None:
        if self.schema != RULE_PREDICATE_SCHEMA:
            raise ResidualIntelligenceError("unsupported rule predicate schema")
        object.__setattr__(self, "feature", _reject_private_name(self.feature, "feature"))
        object.__setattr__(self, "kind", RulePredicateKind(self.kind))
        kind = self.kind
        if kind in {
            RulePredicateKind.PRESENT,
            RulePredicateKind.ABSENT,
            RulePredicateKind.BOOL_TRUE,
            RulePredicateKind.BOOL_FALSE,
        }:
            object.__setattr__(self, "value", "")
            return
        if kind in {RulePredicateKind.IN, RulePredicateKind.NOT_IN}:
            object.__setattr__(
                self,
                "value",
                tuple(
                    _reject_private_name(item, "predicate value")
                    for item in text_tuple(self.value, "predicate value", max_items=64)
                ),
            )
            return
        if kind in {
            RulePredicateKind.INT_EQUALS,
            RulePredicateKind.INT_AT_LEAST,
            RulePredicateKind.INT_AT_MOST,
        }:
            object.__setattr__(
                self,
                "value",
                bounded_int(
                    self.value,
                    "predicate value",
                    minimum=-MAX_FEATURE_VALUE,
                    maximum=MAX_FEATURE_VALUE,
                ),
            )
            return
        if type(self.value) is bool or type(self.value) is int:
            object.__setattr__(self, "value", self.value)
            return
        object.__setattr__(
            self, "value", _reject_private_name(str(self.value or ""), "predicate value")
        )

    @property
    def predicate_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def matches(self, features: Mapping[str, Any]) -> bool:
        observed = features[self.feature] if self.feature in features else _MISSING
        kind = self.kind
        if kind is RulePredicateKind.PRESENT:
            return observed is not _MISSING
        if kind is RulePredicateKind.ABSENT:
            return observed is _MISSING
        if observed is _MISSING:
            return False
        if kind is RulePredicateKind.BOOL_TRUE:
            return observed is True
        if kind is RulePredicateKind.BOOL_FALSE:
            return observed is False
        if kind is RulePredicateKind.EQUALS:
            return type(observed) is type(self.value) and observed == self.value
        if kind is RulePredicateKind.NOT_EQUALS:
            return type(observed) is not type(self.value) or observed != self.value
        if kind is RulePredicateKind.IN:
            return observed in self.value
        if kind is RulePredicateKind.NOT_IN:
            return observed not in self.value
        if type(observed) is not int:
            return False
        if kind is RulePredicateKind.INT_EQUALS:
            return observed == self.value
        if kind is RulePredicateKind.INT_AT_LEAST:
            return observed >= self.value
        if kind is RulePredicateKind.INT_AT_MOST:
            return observed <= self.value
        return False

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: Any = self.value
        if isinstance(value, tuple):
            value = list(value)
        result: dict[str, Any] = {
            "schema": self.schema,
            "feature": self.feature,
            "kind": self.kind.value,
            "value": value,
        }
        if include_id:
            result["predicate_id"] = self.predicate_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RulePredicate:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"predicate_id", "value"},
            noun="rule predicate",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            feature=str(payload.get("feature") or ""),
            kind=RulePredicateKind(str(payload.get("kind") or "")),
            value=payload.get("value", ""),
        )
        claimed = str(payload.get("predicate_id") or "")
        if claimed and claimed != result.predicate_id:
            raise ResidualIntelligenceError("rule predicate identity mismatch")
        return result


@dataclass(frozen=True)
class DeclarativeRule:
    """Priority-ordered conjunctive rule over compact features."""

    rule_id: str
    priority: int
    predicates: tuple[RulePredicate, ...]
    output_class: str
    structured_payload: Mapping[str, Any]
    score_ppm: int
    evidence_references: tuple[str, ...] = ()
    schema: str = DECLARATIVE_RULE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "identity",
            "rule_id",
            "priority",
            "predicates",
            "output_class",
            "structured_payload",
            "score_ppm",
            "evidence_references",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != DECLARATIVE_RULE_SCHEMA:
            raise ResidualIntelligenceError("unsupported declarative rule schema")
        object.__setattr__(self, "rule_id", required_text(self.rule_id, "rule_id", max_bytes=256))
        object.__setattr__(
            self,
            "priority",
            bounded_int(self.priority, "priority", minimum=0, maximum=MAX_SCORE_PPM),
        )
        predicates = tuple(self.predicates)
        if not predicates:
            raise ResidualIntelligenceError("declarative rule requires at least one predicate")
        if len(predicates) > MAX_PREDICATES:
            raise ResidualIntelligenceError(f"declarative rule exceeds {MAX_PREDICATES} predicates")
        if any(not isinstance(item, RulePredicate) for item in predicates):
            raise ResidualIntelligenceError("declarative rule predicates must be typed records")
        object.__setattr__(self, "predicates", predicates)
        object.__setattr__(
            self, "output_class", _reject_private_name(self.output_class, "output_class")
        )
        object.__setattr__(
            self,
            "structured_payload",
            _reject_private_mapping(self.structured_payload, noun="structured_payload"),
        )
        object.__setattr__(self, "score_ppm", _ppm(self.score_ppm, "score_ppm"))
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references", max_items=256),
        )

    @property
    def identity(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def matches(self, features: Mapping[str, Any]) -> bool:
        return all(item.matches(features) for item in self.predicates)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "rule_id": self.rule_id,
            "priority": self.priority,
            "predicates": [item.to_dict() for item in self.predicates],
            "output_class": self.output_class,
            "structured_payload": dict(self.structured_payload),
            "score_ppm": self.score_ppm,
            "evidence_references": list(self.evidence_references),
        }
        if include_id:
            result["identity"] = self.identity
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeclarativeRule:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"identity"},
            noun="declarative rule",
        )
        predicates_payload = payload.get("predicates")
        if isinstance(predicates_payload, (str, bytes, bytearray)) or not isinstance(
            predicates_payload, Sequence
        ):
            raise ResidualIntelligenceError("declarative rule predicates must be a sequence")
        result = cls(
            schema=str(payload.get("schema") or ""),
            rule_id=str(payload.get("rule_id") or ""),
            priority=payload.get("priority"),
            predicates=tuple(RulePredicate.from_dict(item) for item in predicates_payload),
            output_class=str(payload.get("output_class") or ""),
            structured_payload=payload.get("structured_payload") or {},
            score_ppm=payload.get("score_ppm"),
            evidence_references=tuple(payload.get("evidence_references") or ()),
        )
        claimed = str(payload.get("identity") or "")
        if claimed and claimed != result.identity:
            raise ResidualIntelligenceError("declarative rule identity mismatch")
        return result


@dataclass(frozen=True)
class ProcedureBinding:
    """Verified-procedure answer used only when declared preconditions hold."""

    procedure_root: str
    output_class: str
    structured_payload: Mapping[str, Any]
    evidence_references: tuple[str, ...] = ()
    score_ppm: int = MAX_SCORE_PPM
    schema: str = PROCEDURE_BINDING_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "binding_id",
            "procedure_root",
            "output_class",
            "structured_payload",
            "evidence_references",
            "score_ppm",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != PROCEDURE_BINDING_SCHEMA:
            raise ResidualIntelligenceError("unsupported procedure binding schema")
        object.__setattr__(
            self, "procedure_root", required_text(self.procedure_root, "procedure_root")
        )
        object.__setattr__(
            self, "output_class", _reject_private_name(self.output_class, "output_class")
        )
        object.__setattr__(
            self,
            "structured_payload",
            _reject_private_mapping(self.structured_payload, noun="structured_payload"),
        )
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references", max_items=256),
        )
        object.__setattr__(self, "score_ppm", _ppm(self.score_ppm, "score_ppm"))

    @property
    def binding_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "procedure_root": self.procedure_root,
            "output_class": self.output_class,
            "structured_payload": dict(self.structured_payload),
            "evidence_references": list(self.evidence_references),
            "score_ppm": self.score_ppm,
        }
        if include_id:
            result["binding_id"] = self.binding_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureBinding:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"binding_id", "score_ppm"},
            noun="procedure binding",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            procedure_root=str(payload.get("procedure_root") or ""),
            output_class=str(payload.get("output_class") or ""),
            structured_payload=payload.get("structured_payload") or {},
            evidence_references=tuple(payload.get("evidence_references") or ()),
            score_ppm=payload.get("score_ppm", MAX_SCORE_PPM),
        )
        claimed = str(payload.get("binding_id") or "")
        if claimed and claimed != result.binding_id:
            raise ResidualIntelligenceError("procedure binding identity mismatch")
        return result


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


def _procedure_entries(values: Any) -> tuple[ProcedureBinding, ...]:
    if values in (None, ()):
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError("procedure bindings must be a sequence")
    if len(values) > MAX_LOOKUP_ENTRIES:
        raise ResidualIntelligenceError("procedure bindings exceed the lookup bound")
    bindings = tuple(
        item if isinstance(item, ProcedureBinding) else ProcedureBinding.from_dict(item)
        for item in values
    )
    roots = [item.procedure_root for item in bindings]
    if len(set(roots)) != len(roots):
        raise ResidualIntelligenceError("procedure bindings contain duplicate roots")
    return bindings


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


@dataclass(frozen=True)
class BaselinePrediction:
    """Candidate-only baseline output with route, ranking, and cost receipt."""

    task_output: ResidualTaskOutput
    route: BaselineRoute
    feature_identity: str
    cost: BaselineCostReceipt
    disposition: ExpertDisposition
    ranking: tuple[RankingItem, ...] = ()
    candidate_only: bool = True
    schema: str = BASELINE_PREDICTION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "prediction_id",
            "task_output",
            "route",
            "feature_identity",
            "cost",
            "disposition",
            "ranking",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != BASELINE_PREDICTION_SCHEMA:
            raise ResidualIntelligenceError("unsupported baseline prediction schema")
        if not isinstance(self.task_output, ResidualTaskOutput):
            raise ResidualIntelligenceError("task_output must be ResidualTaskOutput")
        if not isinstance(self.cost, BaselineCostReceipt):
            raise ResidualIntelligenceError("cost must be BaselineCostReceipt")
        object.__setattr__(self, "route", BaselineRoute(self.route))
        object.__setattr__(self, "disposition", ExpertDisposition(self.disposition))
        object.__setattr__(
            self,
            "feature_identity",
            required_text(self.feature_identity, "feature_identity"),
        )
        object.__setattr__(self, "ranking", _ranking_tuple(self.ranking))
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if self.task_output.candidate_only is not True:
            raise ResidualIntelligenceError("learned outputs must remain candidate_only=true")
        if self.cost.route is not self.route:
            raise ResidualIntelligenceError("cost receipt route must match the prediction route")
        if self.route is BaselineRoute.REJECT_INPUT:
            if self.disposition is not ExpertDisposition.REJECT_INPUT:
                raise ResidualIntelligenceError("reject route requires REJECT_INPUT")
        elif self.route is BaselineRoute.ABSTAIN:
            if self.disposition not in {
                ExpertDisposition.ABSTAIN,
                ExpertDisposition.OUT_OF_DISTRIBUTION,
                ExpertDisposition.CAPABILITY_UNAVAILABLE,
            }:
                raise ResidualIntelligenceError("abstain route requires an abstaining disposition")
        elif self.route in _PRODUCER_ROUTES:
            if self.disposition not in {
                ExpertDisposition.ACCEPT,
                ExpertDisposition.VALIDATION_REQUIRED,
            }:
                raise ResidualIntelligenceError(
                    "producer routes require ACCEPT or VALIDATION_REQUIRED"
                )
        if self.disposition is ExpertDisposition.ACCEPT and self.task_output.abstained:
            raise ResidualIntelligenceError("ACCEPT cannot be abstained")
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
            "route": self.route.value,
            "feature_identity": self.feature_identity,
            "cost": self.cost.to_dict(),
            "disposition": self.disposition.value,
            "ranking": [item.to_dict() for item in self.ranking],
            "candidate_only": True,
        }
        if include_id:
            result["prediction_id"] = self.prediction_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BaselinePrediction:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"prediction_id", "ranking"},
            noun="baseline prediction",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_output=ResidualTaskOutput.from_dict(payload.get("task_output") or {}),
            route=BaselineRoute(str(payload.get("route") or "")),
            feature_identity=str(payload.get("feature_identity") or ""),
            cost=BaselineCostReceipt.from_dict(payload.get("cost") or {}),
            disposition=ExpertDisposition(str(payload.get("disposition") or "")),
            ranking=tuple(payload.get("ranking") or ()),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("prediction_id") or "")
        if claimed and claimed != result.prediction_id:
            raise ResidualIntelligenceError("baseline prediction identity mismatch")
        return result


@dataclass(frozen=True)
class BaselineEvaluationCase:
    """One labelled evaluation row; expected class is the denominator atom."""

    task_input: ResidualTaskInput
    expected_output_class: str
    critical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("evaluation case requires ResidualTaskInput")
        object.__setattr__(
            self,
            "expected_output_class",
            required_text(self.expected_output_class, "expected_output_class"),
        )
        object.__setattr__(self, "critical", _require_bool(self.critical, "critical"))


@dataclass(frozen=True)
class BaselineEvaluation:
    """Held-out baseline metrics with complete count denominators."""

    example_count: int
    exact_lookup_count: int
    procedure_count: int
    rule_count: int
    ranking_count: int
    linear_count: int
    accept_count: int
    abstain_count: int
    reject_input_count: int
    validation_required_count: int
    false_accept_count: int
    critical_false_accept_count: int
    abstain_route_count: int
    reject_route_count: int
    avoided_model_calls: int
    avoided_remote_calls: int
    model_calls: int
    provider_invocations: int
    remote_input_tokens: int
    remote_output_tokens: int
    cost_microunits: int
    coverage_ppm: int
    precision_ppm: int
    abstention_rate_ppm: int
    schema: str = BASELINE_EVALUATION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "evaluation_id",
            "example_count",
            "exact_lookup_count",
            "procedure_count",
            "rule_count",
            "ranking_count",
            "linear_count",
            "accept_count",
            "abstain_count",
            "reject_input_count",
            "validation_required_count",
            "false_accept_count",
            "critical_false_accept_count",
            "abstain_route_count",
            "reject_route_count",
            "avoided_model_calls",
            "avoided_remote_calls",
            "model_calls",
            "provider_invocations",
            "remote_input_tokens",
            "remote_output_tokens",
            "cost_microunits",
            "coverage_ppm",
            "precision_ppm",
            "abstention_rate_ppm",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != BASELINE_EVALUATION_SCHEMA:
            raise ResidualIntelligenceError("unsupported baseline evaluation schema")
        counts: dict[str, int] = {}
        for field in (
            "example_count",
            "exact_lookup_count",
            "procedure_count",
            "rule_count",
            "ranking_count",
            "linear_count",
            "accept_count",
            "abstain_count",
            "reject_input_count",
            "validation_required_count",
            "false_accept_count",
            "critical_false_accept_count",
            "abstain_route_count",
            "reject_route_count",
            "avoided_model_calls",
            "avoided_remote_calls",
            "model_calls",
            "provider_invocations",
            "remote_input_tokens",
            "remote_output_tokens",
            "cost_microunits",
        ):
            counts[field] = bounded_int(
                getattr(self, field),
                field,
                minimum=0,
                maximum=1_000_000_000_000,
            )
            object.__setattr__(self, field, counts[field])
        n_examples = counts["example_count"]
        if n_examples < 1:
            raise ResidualIntelligenceError("evaluation requires at least one example")
        disposition_total = (
            counts["accept_count"]
            + counts["abstain_count"]
            + counts["reject_input_count"]
            + counts["validation_required_count"]
        )
        if disposition_total != n_examples:
            raise ResidualIntelligenceError(
                "disposition counts must equal the evaluation example population"
            )
        route_total = (
            counts["exact_lookup_count"]
            + counts["procedure_count"]
            + counts["rule_count"]
            + counts["ranking_count"]
            + counts["linear_count"]
            + counts["abstain_route_count"]
            + counts["reject_route_count"]
        )
        if route_total != n_examples:
            raise ResidualIntelligenceError(
                "route counts must equal the evaluation example population"
            )
        if counts["false_accept_count"] > counts["accept_count"]:
            raise ResidualIntelligenceError("false accepts cannot exceed accepts")
        if counts["critical_false_accept_count"] > counts["false_accept_count"]:
            raise ResidualIntelligenceError("critical false accepts cannot exceed false accepts")
        if counts["model_calls"] != 0 or counts["provider_invocations"] != 0:
            raise ResidualIntelligenceError("baseline evaluation cannot include a model call")
        if counts["remote_input_tokens"] != 0 or counts["remote_output_tokens"] != 0:
            raise ResidualIntelligenceError("baseline evaluation cannot include remote tokens")
        if counts["avoided_model_calls"] != n_examples:
            raise ResidualIntelligenceError("every baseline example must avoid a model call")
        if counts["avoided_remote_calls"] != n_examples:
            raise ResidualIntelligenceError("every baseline example must avoid a remote call")
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

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "example_count": self.example_count,
            "exact_lookup_count": self.exact_lookup_count,
            "procedure_count": self.procedure_count,
            "rule_count": self.rule_count,
            "ranking_count": self.ranking_count,
            "linear_count": self.linear_count,
            "accept_count": self.accept_count,
            "abstain_count": self.abstain_count,
            "reject_input_count": self.reject_input_count,
            "validation_required_count": self.validation_required_count,
            "false_accept_count": self.false_accept_count,
            "critical_false_accept_count": self.critical_false_accept_count,
            "abstain_route_count": self.abstain_route_count,
            "reject_route_count": self.reject_route_count,
            "avoided_model_calls": self.avoided_model_calls,
            "avoided_remote_calls": self.avoided_remote_calls,
            "model_calls": 0,
            "provider_invocations": 0,
            "remote_input_tokens": 0,
            "remote_output_tokens": 0,
            "cost_microunits": self.cost_microunits,
            "coverage_ppm": self.coverage_ppm,
            "precision_ppm": self.precision_ppm,
            "abstention_rate_ppm": self.abstention_rate_ppm,
        }
        if include_id:
            result["evaluation_id"] = self.evaluation_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BaselineEvaluation:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"evaluation_id"},
            noun="baseline evaluation",
        )
        kwargs = {
            key: payload.get(key)
            for key in cls._FIELDS
            if key not in {"evaluation_id", "schema"}
        }
        result = cls(schema=str(payload.get("schema") or ""), **kwargs)  # type: ignore[arg-type]
        claimed = str(payload.get("evaluation_id") or "")
        if claimed and claimed != result.evaluation_id:
            raise ResidualIntelligenceError("baseline evaluation identity mismatch")
        return result


def _disposition_for(route: BaselineRoute, *, proposal: bool, abstained: bool) -> ExpertDisposition:
    if route is BaselineRoute.REJECT_INPUT:
        return ExpertDisposition.REJECT_INPUT
    if abstained or route is BaselineRoute.ABSTAIN:
        return ExpertDisposition.ABSTAIN
    if proposal:
        return ExpertDisposition.VALIDATION_REQUIRED
    return ExpertDisposition.ACCEPT


def evaluate_predictions(
    cases: Sequence[BaselineEvaluationCase],
    predictions: Sequence[BaselinePrediction],
) -> BaselineEvaluation:
    if len(cases) != len(predictions):
        raise ResidualIntelligenceError("evaluation cases and predictions must align")
    if not cases:
        raise ResidualIntelligenceError("evaluation requires at least one example")
    exact = procedure = rule = ranking = linear = 0
    accept = abstain = reject_input = validation_required = 0
    false_accept = critical_false_accept = 0
    abstain_route = reject_route = 0
    cost_microunits = 0
    for case, prediction in zip(cases, predictions):
        if prediction.cost.model_calls or prediction.cost.provider_invocations:
            raise ResidualIntelligenceError("baseline evaluation cannot include a model call")
        cost_microunits += prediction.cost.cost_microunits
        route = prediction.route
        if route is BaselineRoute.EXACT_LOOKUP:
            exact += 1
        elif route is BaselineRoute.VERIFIED_PROCEDURE:
            procedure += 1
        elif route is BaselineRoute.DETERMINISTIC_RULE:
            rule += 1
        elif route is BaselineRoute.DETERMINISTIC_RANKING:
            ranking += 1
        elif route is BaselineRoute.LINEAR_LOGISTIC:
            linear += 1
        elif route is BaselineRoute.ABSTAIN:
            abstain_route += 1
        elif route is BaselineRoute.REJECT_INPUT:
            reject_route += 1
        else:
            raise ResidualIntelligenceError("evaluation encountered an unknown baseline route")
        disposition = prediction.disposition
        if disposition is ExpertDisposition.ACCEPT:
            accept += 1
        elif disposition is ExpertDisposition.REJECT_INPUT:
            reject_input += 1
        elif disposition is ExpertDisposition.VALIDATION_REQUIRED:
            validation_required += 1
        else:
            abstain += 1
        accepted = disposition is ExpertDisposition.ACCEPT
        wrong = prediction.task_output.output_class != case.expected_output_class
        if accepted and (wrong or case.critical):
            false_accept += 1
            if case.critical:
                critical_false_accept += 1
    n_examples = len(cases)
    return BaselineEvaluation(
        example_count=n_examples,
        exact_lookup_count=exact,
        procedure_count=procedure,
        rule_count=rule,
        ranking_count=ranking,
        linear_count=linear,
        accept_count=accept,
        abstain_count=abstain,
        reject_input_count=reject_input,
        validation_required_count=validation_required,
        false_accept_count=false_accept,
        critical_false_accept_count=critical_false_accept,
        abstain_route_count=abstain_route,
        reject_route_count=reject_route,
        avoided_model_calls=n_examples,
        avoided_remote_calls=n_examples,
        model_calls=0,
        provider_invocations=0,
        remote_input_tokens=0,
        remote_output_tokens=0,
        cost_microunits=cost_microunits,
        coverage_ppm=(accept * MAX_SCORE_PPM) // n_examples,
        precision_ppm=(
            0 if accept == 0 else ((accept - false_accept) * MAX_SCORE_PPM) // accept
        ),
        abstention_rate_ppm=(abstain * MAX_SCORE_PPM) // n_examples,
    )


def _rank_candidates(
    compact_features: Mapping[str, Any],
    ranking_weights: Mapping[str, int],
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
        score = _clamp_ppm(signal + int(ranking_weights.get(reference, 0)))
        items.append(RankingItem(reference_id=reference, score_ppm=score))
    return tuple(sorted(items, key=lambda item: (-item.score_ppm, item.reference_id)))


@dataclass(frozen=True)
class DeterministicResidualExpert:
    """Exact lookup, procedure, rule, and ranking producer.  Never a model."""

    task_family: ResidualTaskFamily
    calibration_group: str
    feature_names: tuple[str, ...]
    lookup: tuple[ExactLookupEntry, ...] = ()
    rules: tuple[DeclarativeRule, ...] = ()
    procedures: tuple[ProcedureBinding, ...] = ()
    ranking_weights: tuple[tuple[str, int], ...] = ()
    schema: str = DETERMINISTIC_EXPERT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "expert_version",
            "task_family",
            "calibration_group",
            "feature_names",
            "lookup",
            "rules",
            "procedures",
            "ranking_weights",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != DETERMINISTIC_EXPERT_SCHEMA:
            raise ResidualIntelligenceError("unsupported deterministic expert schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(
            self,
            "calibration_group",
            required_text(self.calibration_group, "calibration_group"),
        )
        object.__setattr__(self, "feature_names", _feature_names(self.feature_names))
        object.__setattr__(self, "lookup", _lookup_entries(self.lookup))
        object.__setattr__(self, "rules", _rule_entries(self.rules))
        object.__setattr__(self, "procedures", _procedure_entries(self.procedures))
        object.__setattr__(self, "ranking_weights", _weight_pairs(self.ranking_weights))

    @property
    def expert_version(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def lookup_by_identity(self) -> dict[str, ExactLookupEntry]:
        return {item.feature_identity: item for item in self.lookup}

    def procedure_by_root(self) -> dict[str, ProcedureBinding]:
        return {item.procedure_root: item for item in self.procedures}

    def ranking_weight_map(self) -> dict[str, int]:
        return {key: value for key, value in self.ranking_weights}

    def predict(self, task_input: ResidualTaskInput) -> BaselinePrediction:
        return self._predict(task_input, allow_ranking=True)

    def _predict(
        self, task_input: ResidualTaskInput, *, allow_ranking: bool
    ) -> BaselinePrediction:
        if not isinstance(task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("predict requires ResidualTaskInput")
        features = dict(task_input.compact_features)
        feature_identity = stable_feature_identity(features, self.feature_names)
        feature_ops = len(self.feature_names) + 4
        if task_input.task_family is not self.task_family:
            return self._emit(
                task_input,
                route=BaselineRoute.REJECT_INPUT,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
                reasons=(REASON_REJECT_INPUT, REASON_FAMILY_MISMATCH),
            )
        if FEATURE_INPUT_VALID in features and features[FEATURE_INPUT_VALID] is not True:
            if features[FEATURE_INPUT_VALID] is not False:
                raise ResidualIntelligenceError("input_valid must be boolean")
            return self._emit(
                task_input,
                route=BaselineRoute.REJECT_INPUT,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
                reasons=(REASON_REJECT_INPUT,),
            )
        if features.get(FEATURE_CRITICAL_BOUNDARY) is True:
            return self._emit(
                task_input,
                route=BaselineRoute.ABSTAIN,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
                reasons=(REASON_CRITICAL_BOUNDARY,),
            )
        if features.get(FEATURE_CRITICAL_BOUNDARY) not in (None, False):
            raise ResidualIntelligenceError("critical_boundary must be boolean")

        hit = self.lookup_by_identity().get(feature_identity)
        if hit is not None:
            return self._emit(
                task_input,
                route=BaselineRoute.EXACT_LOOKUP,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
                output_class=hit.output_class,
                payload=hit.structured_payload,
                score_ppm=hit.score_ppm,
                evidence=hit.evidence_references,
                reasons=(REASON_EXACT_LOOKUP,),
            )

        procedure_requested = bool(
            features.get(FEATURE_PROCEDURE_ROOT)
            or features.get(FEATURE_PROCEDURE_ANSWER) is True
        )
        if procedure_requested:
            preconditions_ok = features.get(FEATURE_PROCEDURE_PRECONDITIONS) is True
            if not preconditions_ok:
                return self._emit(
                    task_input,
                    route=BaselineRoute.ABSTAIN,
                    feature_identity=feature_identity,
                    feature_ops=feature_ops,
                    reasons=(REASON_PROCEDURE_PRECONDITION,),
                )
            root = features.get(FEATURE_PROCEDURE_ROOT)
            binding = self.procedure_by_root().get(str(root or ""))
            if binding is None:
                return self._emit(
                    task_input,
                    route=BaselineRoute.ABSTAIN,
                    feature_identity=feature_identity,
                    feature_ops=feature_ops,
                    reasons=(REASON_PROCEDURE_UNBOUND,),
                )
            return self._emit(
                task_input,
                route=BaselineRoute.VERIFIED_PROCEDURE,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
                output_class=binding.output_class,
                payload=binding.structured_payload,
                score_ppm=binding.score_ppm,
                evidence=binding.evidence_references,
                reasons=(REASON_VERIFIED_PROCEDURE,),
            )

        for rule in self.rules:
            if rule.matches(features):
                return self._emit(
                    task_input,
                    route=BaselineRoute.DETERMINISTIC_RULE,
                    feature_identity=feature_identity,
                    feature_ops=feature_ops,
                    output_class=rule.output_class,
                    payload=rule.structured_payload,
                    score_ppm=rule.score_ppm,
                    evidence=rule.evidence_references + (f"rule:{rule.rule_id}",),
                    reasons=(REASON_DETERMINISTIC_RULE,),
                )

        if allow_ranking:
            ranking = _rank_candidates(features, self.ranking_weight_map())
            if ranking:
                return self._emit(
                    task_input,
                    route=BaselineRoute.DETERMINISTIC_RANKING,
                    feature_identity=feature_identity,
                    feature_ops=feature_ops,
                    output_class=task_input.task_family.value,
                    payload={
                        "ranked_reference_ids": [item.reference_id for item in ranking],
                        "scores_ppm": [item.score_ppm for item in ranking],
                    },
                    score_ppm=ranking[0].score_ppm,
                    evidence=("ranking:stable-sort",),
                    reasons=(REASON_DETERMINISTIC_RANKING,),
                    ranking=ranking,
                )

        return self._emit(
            task_input,
            route=BaselineRoute.ABSTAIN,
            feature_identity=feature_identity,
            feature_ops=feature_ops,
            reasons=(REASON_NO_DETERMINISTIC_MATCH,),
        )

    def evaluate(
        self, cases: Sequence[BaselineEvaluationCase]
    ) -> BaselineEvaluation:
        predictions = tuple(self.predict(case.task_input) for case in cases)
        return evaluate_predictions(cases, predictions)

    def _emit(
        self,
        task_input: ResidualTaskInput,
        *,
        route: BaselineRoute,
        feature_identity: str,
        feature_ops: int,
        reasons: tuple[str, ...],
        output_class: str = ABSTAIN_OUTPUT_CLASS,
        payload: Mapping[str, Any] | None = None,
        score_ppm: int = 0,
        evidence: tuple[str, ...] = (),
        ranking: tuple[RankingItem, ...] = (),
    ) -> BaselinePrediction:
        abstained = route in {BaselineRoute.ABSTAIN, BaselineRoute.REJECT_INPUT}
        proposal = (not abstained) and task_input.risk_class in PROPOSAL_RISKS
        chosen_class = ABSTAIN_OUTPUT_CLASS if abstained else output_class
        structured = {} if abstained else dict(payload or {})
        reason_codes = reasons
        if proposal and REASON_VALIDATION_REQUIRED not in reason_codes:
            reason_codes = reasons + (REASON_VALIDATION_REQUIRED, REASON_R4_R5_PROPOSAL)
        if chosen_class not in task_input.allowed_outputs:
            if ABSTAIN_OUTPUT_CLASS not in task_input.allowed_outputs:
                raise ResidualIntelligenceError("ABSTAIN must be in allowed_outputs")
            abstained = True
            route = BaselineRoute.ABSTAIN if route is not BaselineRoute.REJECT_INPUT else route
            chosen_class = ABSTAIN_OUTPUT_CLASS
            structured = {}
            reason_codes = (REASON_OUTPUT_NOT_ALLOWED,) + reason_codes
            proposal = False
        if abstained and ABSTAIN_OUTPUT_CLASS not in task_input.allowed_outputs:
            raise ResidualIntelligenceError("ABSTAIN must be in allowed_outputs")
        output = ResidualTaskOutput(
            output_class=chosen_class,
            structured_payload=structured,
            confidence_or_score=_ppm(score_ppm if not abstained else 0, "confidence_or_score"),
            calibration_group=self.calibration_group,
            abstained=abstained,
            reason_codes=reason_codes,
            evidence_references=() if abstained else evidence,
            candidate_only=True,
        )
        disposition = _disposition_for(route, proposal=proposal, abstained=abstained)
        return BaselinePrediction(
            task_output=output,
            route=route,
            feature_identity=feature_identity,
            cost=_local_cost(route, feature_ops=feature_ops),
            disposition=disposition,
            ranking=ranking,
            candidate_only=True,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "calibration_group": self.calibration_group,
            "feature_names": list(self.feature_names),
            "lookup": [item.to_dict() for item in self.lookup],
            "rules": [item.to_dict() for item in self.rules],
            "procedures": [item.to_dict() for item in self.procedures],
            "ranking_weights": [[key, value] for key, value in self.ranking_weights],
        }
        if include_id:
            result["expert_version"] = self.expert_version
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeterministicResidualExpert:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {"expert_version", "lookup", "rules", "procedures", "ranking_weights"},
            noun="deterministic residual expert",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            calibration_group=str(payload.get("calibration_group") or ""),
            feature_names=tuple(payload.get("feature_names") or ()),
            lookup=tuple(payload.get("lookup") or ()),
            rules=tuple(payload.get("rules") or ()),
            procedures=tuple(payload.get("procedures") or ()),
            ranking_weights=tuple(payload.get("ranking_weights") or ()),
        )
        claimed = str(payload.get("expert_version") or "")
        if claimed and claimed != result.expert_version:
            raise ResidualIntelligenceError("deterministic expert version mismatch")
        return result


@dataclass(frozen=True)
class LinearResidualExpert:
    """Bounded integer linear/logistic scorer after the deterministic cascade."""

    deterministic: DeterministicResidualExpert
    form: LinearForm
    feature_names: tuple[str, ...]
    class_labels: tuple[str, ...]
    coefficients: tuple[tuple[int, ...], ...] = ()
    intercepts: tuple[int, ...] = ()
    threshold_ppm: int = 500_000
    fitted: bool = False
    admission_id: str = ""
    checkpoint_count: int = 0
    schema: str = LINEAR_EXPERT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "expert_version",
            "deterministic",
            "form",
            "feature_names",
            "class_labels",
            "coefficients",
            "intercepts",
            "threshold_ppm",
            "fitted",
            "admission_id",
            "checkpoint_count",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != LINEAR_EXPERT_SCHEMA:
            raise ResidualIntelligenceError("unsupported linear expert schema")
        if not isinstance(self.deterministic, DeterministicResidualExpert):
            raise ResidualIntelligenceError(
                "linear expert requires a typed DeterministicResidualExpert"
            )
        object.__setattr__(self, "form", LinearForm(self.form))
        object.__setattr__(self, "feature_names", _feature_names(self.feature_names))
        labels = text_tuple(self.class_labels, "class_labels", max_items=256)
        if any(item == ABSTAIN_OUTPUT_CLASS for item in labels):
            raise ResidualIntelligenceError("class_labels cannot include ABSTAIN")
        object.__setattr__(self, "class_labels", labels)
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
        object.__setattr__(self, "threshold_ppm", _ppm(self.threshold_ppm, "threshold_ppm"))
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
                maximum=MAX_FIT_CHECKPOINTS,
            ),
        )
        if self.fitted and not self.admission_id:
            raise ResidualIntelligenceError("fitted linear baseline requires an admission_id")
        if self.fitted and self.checkpoint_count != 1:
            raise ResidualIntelligenceError("fitted linear baseline permits exactly one checkpoint")
        if not labels and (self.coefficients or self.intercepts):
            raise ResidualIntelligenceError("coefficients require class_labels")

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
            if self.form is LinearForm.LOGISTIC:
                scores.append(logistic_ppm(raw))
            else:
                scores.append(_clamp_ppm(raw))
        return tuple(scores)

    def predict(self, task_input: ResidualTaskInput) -> BaselinePrediction:
        prior = self.deterministic._predict(task_input, allow_ranking=True)
        if prior.route is not BaselineRoute.ABSTAIN:
            return prior
        reasons = prior.task_output.reason_codes
        if any(
            code in reasons
            for code in (
                REASON_CRITICAL_BOUNDARY,
                REASON_PROCEDURE_PRECONDITION,
                REASON_PROCEDURE_UNBOUND,
                REASON_REJECT_INPUT,
                REASON_FAMILY_MISMATCH,
            )
        ):
            return prior
        feature_identity = prior.feature_identity
        feature_ops = prior.cost.feature_ops + len(self.feature_names)
        if not self.coefficients_available:
            return self.deterministic._emit(
                task_input,
                route=BaselineRoute.ABSTAIN,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
                reasons=(REASON_LINEAR_UNAVAILABLE,) + reasons,
            )
        vector = extract_linear_vector(task_input.compact_features, self.feature_names)
        if vector is None:
            return self.deterministic._emit(
                task_input,
                route=BaselineRoute.ABSTAIN,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
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
            return self.deterministic._emit(
                task_input,
                route=BaselineRoute.ABSTAIN,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
                reasons=(REASON_LINEAR_NO_SIGNAL,),
            )
        best = ranked[0]
        if len(self.class_labels) == 1 and best.score_ppm < self.threshold_ppm:
            return self.deterministic._emit(
                task_input,
                route=BaselineRoute.ABSTAIN,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
                reasons=(REASON_LINEAR_NO_SIGNAL,),
                ranking=ranked,
            )
        if best.score_ppm <= 0 and self.form is LinearForm.LINEAR:
            return self.deterministic._emit(
                task_input,
                route=BaselineRoute.ABSTAIN,
                feature_identity=feature_identity,
                feature_ops=feature_ops,
                reasons=(REASON_LINEAR_NO_SIGNAL,),
                ranking=ranked,
            )
        payload = {"label": best.reference_id, "reference_ids": []}
        if task_input.task_family is ResidualTaskFamily.FAILURE_ATTRIBUTION:
            payload = {
                "failure_class": best.reference_id,
                "recommended_action": "expand_context_reference",
                "reference_ids": [],
            }
        return self.deterministic._emit(
            task_input,
            route=BaselineRoute.LINEAR_LOGISTIC,
            feature_identity=feature_identity,
            feature_ops=feature_ops,
            output_class=task_input.task_family.value,
            payload=payload,
            score_ppm=best.score_ppm,
            evidence=(f"linear:{self.form.value}",),
            reasons=(REASON_LINEAR_LOGISTIC,),
            ranking=ranked,
        )

    def evaluate(
        self, cases: Sequence[BaselineEvaluationCase]
    ) -> BaselineEvaluation:
        predictions = tuple(self.predict(case.task_input) for case in cases)
        return evaluate_predictions(cases, predictions)

    def fit(
        self,
        *,
        admission: TrainingCorpusAdmission,
        cases: Sequence[BaselineEvaluationCase],
        cpu_seconds: int = 0,
    ) -> LinearResidualExpert:
        if not isinstance(admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("fit requires TrainingCorpusAdmission")
        if admission.admission_decision is not TrainingAvailability.ADMITTED:
            raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)
        admission.require_training_admitted()
        if not admission.can_train:
            raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)
        bounded_int(cpu_seconds, "cpu_seconds", minimum=0, maximum=MAX_FIT_CPU_SECONDS)
        if len(cases) > MAX_LINEAR_EXAMPLES:
            raise ResidualIntelligenceError(
                f"linear fit exceeds {MAX_LINEAR_EXAMPLES} examples"
            )
        if not cases:
            raise ResidualIntelligenceError("linear fit requires at least one example")
        ordered = tuple(
            sorted(
                cases,
                key=lambda item: (item.task_input.input_id, item.expected_output_class),
            )
        )
        rows: list[tuple[tuple[int, ...], str]] = []
        for case in ordered:
            vector = extract_linear_vector(case.task_input.compact_features, self.feature_names)
            if vector is None:
                raise ResidualIntelligenceError("linear fit is missing a stable feature")
            rows.append((vector, case.expected_output_class))
        labels = self.class_labels or tuple(
            sorted({label for _vector, label in rows if label != ABSTAIN_OUTPUT_CLASS})
        )
        if not labels:
            raise ResidualIntelligenceError("linear fit requires a non-abstain class label")
        n_features = len(self.feature_names)
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
            prior_den = len(rows)
            intercept = 0
            if 0 < prior_num < prior_den:
                intercept = ((2 * prior_num - prior_den) * LOGIT_SCALE) // prior_den
            intercepts.append(
                max(-MAX_COEFFICIENT, min(MAX_COEFFICIENT, intercept))
            )
            coefficients.append(tuple(row))
        return LinearResidualExpert(
            schema=self.schema,
            deterministic=self.deterministic,
            form=self.form,
            feature_names=self.feature_names,
            class_labels=labels,
            coefficients=tuple(coefficients),
            intercepts=tuple(intercepts),
            threshold_ppm=self.threshold_ppm,
            fitted=True,
            admission_id=admission.admission_id,
            checkpoint_count=1,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "deterministic": self.deterministic.to_dict(),
            "form": self.form.value,
            "feature_names": list(self.feature_names),
            "class_labels": list(self.class_labels),
            "coefficients": [list(row) for row in self.coefficients],
            "intercepts": list(self.intercepts),
            "threshold_ppm": self.threshold_ppm,
            "fitted": self.fitted,
            "admission_id": self.admission_id,
            "checkpoint_count": self.checkpoint_count,
        }
        if include_id:
            result["expert_version"] = self.expert_version
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LinearResidualExpert:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {
                "expert_version",
                "coefficients",
                "intercepts",
                "threshold_ppm",
                "fitted",
                "admission_id",
                "checkpoint_count",
            },
            noun="linear residual expert",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            deterministic=DeterministicResidualExpert.from_dict(
                payload.get("deterministic") or {}
            ),
            form=LinearForm(str(payload.get("form") or "")),
            feature_names=tuple(payload.get("feature_names") or ()),
            class_labels=tuple(payload.get("class_labels") or ()),
            coefficients=tuple(tuple(row) for row in (payload.get("coefficients") or ())),
            intercepts=tuple(payload.get("intercepts") or ()),
            threshold_ppm=payload.get("threshold_ppm", 500_000),
            fitted=payload.get("fitted", False),
            admission_id=str(payload.get("admission_id") or ""),
            checkpoint_count=payload.get("checkpoint_count", 0),
        )
        claimed = str(payload.get("expert_version") or "")
        if claimed and claimed != result.expert_version:
            raise ResidualIntelligenceError("linear expert version mismatch")
        return result


__all__ = (
    "BASELINE_CASCADE_ORDER",
    "BaselineCostReceipt",
    "BaselineEvaluation",
    "BaselineEvaluationCase",
    "BaselinePrediction",
    "BaselineRoute",
    "DeclarativeRule",
    "DeterministicResidualExpert",
    "ExactLookupEntry",
    "LinearForm",
    "LinearResidualExpert",
    "ProcedureBinding",
    "RankingItem",
    "RulePredicate",
    "RulePredicateKind",
    "evaluate_predictions",
    "extract_stable_features",
    "logistic_ppm",
    "stable_feature_identity",
)
