"""Bounded desired-vs-observed AND/OR obligation graph compilation.

The compiler in this module is deliberately symbolic.  It consumes typed
predicates, observed facts, reviewed producer rules, and body-free references;
it never derives proof semantics from prose.  Backward chaining turns each
desired predicate into an obligation, alternative producers into an ``OR``
refinement, and each producer's required premises into an ``AND`` refinement.

Unknown or unsupported semantics remain explicit review obligations.  A graph
is suitable for candidate generation only when it is finite and consistent
and every open leaf is named by a task candidate.  This is a planning
admission result, not proof or completion authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Final


OBLIGATION_GRAPH_INTERFACE: Final[str] = "ObligationGraph@1"
OBLIGATION_GRAPH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/obligation-graph@1"
)
TYPED_PREDICATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/obligation-predicate@1"
)
TYPED_INTENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/typed-obligation-intent@1"
)
OBSERVED_FACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/observed-obligation-fact@1"
)
PRODUCER_RULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/obligation-producer-rule@1"
)
ASSUMPTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/obligation-assumption@1"
)
INVALIDATION_SELECTOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/obligation-invalidation-selector@1"
)
TASK_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/obligation-task-candidate@1"
)
OBLIGATION_NODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/obligation-node@1"
)
OBLIGATION_REFINEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/obligation-refinement@1"
)
OBLIGATION_ISSUE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/obligation-issue@1"
)

_MAX_TEXT_BYTES = 8 * 1024
_MAX_REFERENCES = 4_096
_IDENTIFIER_RE = re.compile(r"^[^\x00\r\n\t]{1,2048}$")


class ObligationCompilationError(ValueError):
    """An input cannot be represented without weakening graph semantics."""


class ObligationBoundsError(ObligationCompilationError):
    """A declared compilation bound is invalid or has been exceeded."""


class PredicatePolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

    @property
    def opposite(self) -> "PredicatePolarity":
        return (
            PredicatePolarity.NEGATIVE
            if self is PredicatePolarity.POSITIVE
            else PredicatePolarity.POSITIVE
        )


class SemanticSupport(str, Enum):
    REVIEWED = "reviewed"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class FactTruth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    REVIEW = "review"


class FactAuthority(str, Enum):
    CURRENT_ROOT_FACT = "current_root_fact"
    REVIEWED_CONTRACT = "reviewed_contract"
    REVIEWED_POLICY = "reviewed_policy"
    BOUNDED_OBSERVATION = "bounded_observation"
    PROOF_RECEIPT = "proof_receipt"
    COUNTEREXAMPLE = "counterexample"
    DIAGNOSTIC = "diagnostic"
    NOMINATION_ONLY = "nomination_only"

    @property
    def may_discharge(self) -> bool:
        return self not in {
            FactAuthority.DIAGNOSTIC,
            FactAuthority.NOMINATION_ONLY,
        }


class InvalidationSelectorKind(str, Enum):
    PATH = "path"
    SYMBOL = "symbol"
    ROOT = "root"
    EVIDENCE = "evidence"
    POLICY = "policy"
    PROPERTY = "property"
    CAPABILITY = "capability"
    TASK_SOURCE = "task_source"


class AssumptionStatus(str, Enum):
    ACTIVE = "active"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class ObligationNodeKind(str, Enum):
    GOAL = "goal"
    SUBGOAL = "subgoal"
    PRODUCER = "producer"


class RefinementKind(str, Enum):
    AND = "and"
    OR = "or"


class ObligationStatus(str, Enum):
    DISCHARGED = "discharged"
    OPEN = "open"
    REVIEW = "review"
    BLOCKED = "blocked"
    CONTRADICTED = "contradicted"


class ObligationIssueKind(str, Enum):
    CYCLE = "cycle"
    CONTRADICTION = "contradiction"
    UNCOVERED_LEAF = "uncovered_leaf"
    INCONSISTENT_PREMISE = "inconsistent_premise"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"
    INVALID_ASSUMPTION = "invalid_assumption"
    UNKNOWN_ASSUMPTION = "unknown_assumption"
    INVALID_TASK_CLOSURE = "invalid_task_closure"
    BOUND_EXCEEDED = "bound_exceeded"
    STALE_FACT = "stale_fact"
    NON_AUTHORITATIVE_FACT = "non_authoritative_fact"
    INCOMPLETE_QUERY_PLAN = "incomplete_query_plan"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"


class IssueSeverity(str, Enum):
    ERROR = "error"
    REVIEW = "review"
    WARNING = "warning"


class ObligationGraphDecision(str, Enum):
    READY = "ready"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ObligationCompilationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ObligationCompilationError(f"{name} must not be empty")
    if result and not _IDENTIFIER_RE.fullmatch(result):
        raise ObligationCompilationError(f"{name} is not a compact identifier")
    return result


def _text(value: Any, name: str, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ObligationCompilationError(f"{name} must be a string")
    result = " ".join(value.split())
    if required and not result:
        raise ObligationCompilationError(f"{name} must not be empty")
    if "\x00" in result or len(result.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ObligationCompilationError(f"{name} is not bounded UTF-8 text")
    return result


def _items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _ids(
    value: Any,
    name: str,
    *,
    required: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    result: list[str] = []
    for item in _items(value):
        normalized = _identifier(item, name)
        if normalized not in result:
            result.append(normalized)
    if len(result) > _MAX_REFERENCES:
        raise ObligationBoundsError(f"{name} exceeds {_MAX_REFERENCES} entries")
    if required and not result:
        raise ObligationCompilationError(f"{name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _frozen_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ObligationCompilationError(f"{name} must be a mapping")

    def visit(item: Any, depth: int) -> Any:
        if depth > 6:
            raise ObligationBoundsError(f"{name} exceeds nesting bound")
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            raise ObligationCompilationError(f"{name} must not contain floats")
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, Mapping):
            return MappingProxyType(
                {
                    _identifier(str(key), f"{name} key"): visit(member, depth + 1)
                    for key, member in sorted(item.items(), key=lambda pair: str(pair[0]))
                }
            )
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            return tuple(visit(member, depth + 1) for member in item)
        raise ObligationCompilationError(
            f"{name} contains unsupported type {type(item).__name__}"
        )

    result = visit(value, 0)
    assert isinstance(result, Mapping)
    return result


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    raise ObligationCompilationError(
        f"canonical payload contains unsupported type {type(value).__name__}"
    )


def _content_id(namespace: str, value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _enum(value: Any, cls: type[Enum], name: str) -> Any:
    if isinstance(value, cls):
        return value
    raw = getattr(value, "value", value)
    try:
        return cls(str(raw))
    except (TypeError, ValueError) as exc:
        raise ObligationCompilationError(
            f"{name} has unsupported value {value!r}"
        ) from exc


def _decode_record(value: Any, cls: type[Any], name: str) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_dict(value)
    raise ObligationCompilationError(f"{name} must contain {cls.__name__} values")


def _decode_records(
    value: Any,
    cls: type[Any],
    name: str,
    *,
    key: str,
    required: bool = False,
) -> tuple[Any, ...]:
    records = tuple(_decode_record(item, cls, name) for item in _items(value))
    ordered = tuple(sorted(records, key=lambda item: getattr(item, key)))
    keys = [getattr(item, key) for item in ordered]
    if len(keys) != len(set(keys)):
        raise ObligationCompilationError(f"{name} identifiers must be unique")
    if required and not ordered:
        raise ObligationCompilationError(f"{name} must not be empty")
    return ordered


def _verify_claimed_id(payload: Mapping[str, Any], actual: str, *names: str) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise ObligationCompilationError(
                f"{name} does not match canonical content"
            )


def _polarity(value: Any) -> PredicatePolarity:
    if isinstance(value, bool):
        return (
            PredicatePolarity.POSITIVE if value else PredicatePolarity.NEGATIVE
        )
    raw = str(getattr(value, "value", value) or "positive").casefold()
    aliases = {
        "positive": PredicatePolarity.POSITIVE,
        "true": PredicatePolarity.POSITIVE,
        "asserted": PredicatePolarity.POSITIVE,
        "negative": PredicatePolarity.NEGATIVE,
        "false": PredicatePolarity.NEGATIVE,
        "negated": PredicatePolarity.NEGATIVE,
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ObligationCompilationError(
            f"polarity has unsupported value {value!r}"
        ) from exc


def _fact_truth(value: Any) -> FactTruth:
    if isinstance(value, FactTruth):
        return value
    if isinstance(value, bool):
        return FactTruth.TRUE if value else FactTruth.FALSE
    raw = str(getattr(value, "value", value) or "unknown").casefold()
    aliases = {
        "true": FactTruth.TRUE,
        "satisfied": FactTruth.TRUE,
        "observed": FactTruth.TRUE,
        "false": FactTruth.FALSE,
        "contradicted": FactTruth.FALSE,
        "unknown": FactTruth.UNKNOWN,
        "inconclusive": FactTruth.UNKNOWN,
        "review": FactTruth.REVIEW,
        "unsupported": FactTruth.REVIEW,
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ObligationCompilationError(
            f"truth has unsupported value {value!r}"
        ) from exc


@dataclass(frozen=True)
class InvalidationSelector:
    selector_id: str
    kind: InvalidationSelectorKind
    value_ref: str
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "selector_id", _identifier(self.selector_id, "selector_id")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, InvalidationSelectorKind, "kind")
        )
        object.__setattr__(
            self, "value_ref", _identifier(self.value_ref, "value_ref")
        )
        object.__setattr__(
            self,
            "provenance_refs",
            _ids(self.provenance_refs, "provenance_refs"),
        )

    def matches(self, changes: Mapping[str, Any]) -> bool:
        names = {
            InvalidationSelectorKind.PATH: ("paths", "changed_paths"),
            InvalidationSelectorKind.SYMBOL: ("symbols", "changed_symbols"),
            InvalidationSelectorKind.ROOT: ("roots", "changed_roots", "tree_id"),
            InvalidationSelectorKind.EVIDENCE: (
                "evidence_refs",
                "evidence_references",
            ),
            InvalidationSelectorKind.POLICY: ("policy_ids", "policy_id"),
            InvalidationSelectorKind.PROPERTY: ("property_ids",),
            InvalidationSelectorKind.CAPABILITY: ("capability_ids",),
            InvalidationSelectorKind.TASK_SOURCE: (
                "task_source_ids",
                "objective_revision",
            ),
        }[self.kind]
        for name in names:
            values = _items(changes.get(name))
            if self.value_ref in {str(item) for item in values}:
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INVALIDATION_SELECTOR_SCHEMA,
            "selector_id": self.selector_id,
            "kind": self.kind.value,
            "value_ref": self.value_ref,
            "provenance_refs": list(self.provenance_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InvalidationSelector":
        return cls(
            selector_id=payload.get("selector_id", ""),
            kind=payload.get("kind", ""),
            value_ref=payload.get("value_ref", payload.get("value", "")),
            provenance_refs=tuple(payload.get("provenance_refs") or ()),
        )


@dataclass(frozen=True)
class AssumptionBinding:
    assumption_id: str
    statement_ref: str
    provenance_refs: tuple[str, ...]
    invalidation_selectors: tuple[InvalidationSelector, ...]
    status: AssumptionStatus = AssumptionStatus.ACTIVE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "assumption_id", _identifier(self.assumption_id, "assumption_id")
        )
        object.__setattr__(
            self, "statement_ref", _identifier(self.statement_ref, "statement_ref")
        )
        object.__setattr__(
            self,
            "provenance_refs",
            _ids(self.provenance_refs, "provenance_refs", required=True),
        )
        selectors = _decode_records(
            self.invalidation_selectors,
            InvalidationSelector,
            "invalidation_selectors",
            key="selector_id",
            required=True,
        )
        object.__setattr__(self, "invalidation_selectors", selectors)
        object.__setattr__(
            self, "status", _enum(self.status, AssumptionStatus, "status")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ASSUMPTION_SCHEMA,
            "assumption_id": self.assumption_id,
            "statement_ref": self.statement_ref,
            "provenance_refs": list(self.provenance_refs),
            "invalidation_selectors": [
                item.to_dict() for item in self.invalidation_selectors
            ],
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssumptionBinding":
        return cls(
            assumption_id=payload.get("assumption_id", ""),
            statement_ref=payload.get("statement_ref", ""),
            provenance_refs=tuple(payload.get("provenance_refs") or ()),
            invalidation_selectors=tuple(
                payload.get("invalidation_selectors") or ()
            ),
            status=payload.get("status", AssumptionStatus.ACTIVE.value),
        )


@dataclass(frozen=True)
class TypedPredicate:
    """A semantic atom identified independently from human-readable text."""

    predicate_id: str
    predicate_type: str
    subject_ref: str
    object_ref: str = ""
    polarity: PredicatePolarity = PredicatePolarity.POSITIVE
    support: SemanticSupport = SemanticSupport.REVIEWED
    property_id: str = ""
    provenance_refs: tuple[str, ...] = ()
    assumption_refs: tuple[str, ...] = ()
    invalidation_selectors: tuple[InvalidationSelector, ...] = ()
    proof_requirement_refs: tuple[str, ...] = ()
    validation_requirement_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("predicate_id", "predicate_type", "subject_ref"):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self, "object_ref", _identifier(self.object_ref, "object_ref", required=False)
        )
        object.__setattr__(self, "polarity", _polarity(self.polarity))
        object.__setattr__(
            self, "support", _enum(self.support, SemanticSupport, "support")
        )
        object.__setattr__(
            self, "property_id", _identifier(self.property_id, "property_id", required=False)
        )
        for name in (
            "provenance_refs",
            "assumption_refs",
            "proof_requirement_refs",
            "validation_requirement_refs",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "invalidation_selectors",
            _decode_records(
                self.invalidation_selectors,
                InvalidationSelector,
                "invalidation_selectors",
                key="selector_id",
            ),
        )

    @property
    def semantic_key(self) -> tuple[str, str, str]:
        return (self.predicate_type, self.subject_ref, self.object_ref)

    @property
    def signed_key(self) -> tuple[str, str, str, PredicatePolarity]:
        return (*self.semantic_key, self.polarity)

    def is_opposite(self, other: "TypedPredicate") -> bool:
        return (
            self.semantic_key == other.semantic_key
            and self.polarity is other.polarity.opposite
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TYPED_PREDICATE_SCHEMA,
            "predicate_id": self.predicate_id,
            "predicate_type": self.predicate_type,
            "subject_ref": self.subject_ref,
            "object_ref": self.object_ref,
            "polarity": self.polarity.value,
            "support": self.support.value,
            "property_id": self.property_id,
            "provenance_refs": list(self.provenance_refs),
            "assumption_refs": list(self.assumption_refs),
            "invalidation_selectors": [
                item.to_dict() for item in self.invalidation_selectors
            ],
            "proof_requirement_refs": list(self.proof_requirement_refs),
            "validation_requirement_refs": list(self.validation_requirement_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypedPredicate":
        return cls(
            predicate_id=payload.get("predicate_id", payload.get("id", "")),
            predicate_type=payload.get(
                "predicate_type", payload.get("kind", payload.get("relation", ""))
            ),
            subject_ref=payload.get(
                "subject_ref", payload.get("subject_id", payload.get("subject", ""))
            ),
            object_ref=payload.get(
                "object_ref", payload.get("object_id", payload.get("value_ref", ""))
            ),
            polarity=payload.get("polarity", PredicatePolarity.POSITIVE.value),
            support=payload.get(
                "support",
                SemanticSupport.UNSUPPORTED.value
                if payload.get("unsupported", False)
                else SemanticSupport.REVIEWED.value,
            ),
            property_id=payload.get("property_id", ""),
            provenance_refs=tuple(
                payload.get("provenance_refs")
                or payload.get("source_refs")
                or ()
            ),
            assumption_refs=tuple(payload.get("assumption_refs") or ()),
            invalidation_selectors=tuple(
                payload.get("invalidation_selectors")
                or payload.get("invalidators")
                or ()
            ),
            proof_requirement_refs=tuple(
                payload.get("proof_requirement_refs")
                or payload.get("proof_requirements")
                or ()
            ),
            validation_requirement_refs=tuple(
                payload.get("validation_requirement_refs")
                or payload.get("validation_requirements")
                or ()
            ),
        )


Predicate = TypedPredicate


@dataclass(frozen=True)
class TypedIntent:
    intent_id: str
    desired_predicates: tuple[TypedPredicate, ...]
    source_refs: tuple[str, ...]
    current_root_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _identifier(self.intent_id, "intent_id"))
        object.__setattr__(
            self,
            "desired_predicates",
            _decode_records(
                self.desired_predicates,
                TypedPredicate,
                "desired_predicates",
                key="predicate_id",
                required=True,
            ),
        )
        object.__setattr__(
            self, "source_refs", _ids(self.source_refs, "source_refs", required=True)
        )
        object.__setattr__(
            self,
            "current_root_id",
            _identifier(self.current_root_id, "current_root_id", required=False),
        )
        object.__setattr__(
            self, "metadata", _frozen_mapping(self.metadata, "metadata")
        )

    @property
    def goal_predicate_ids(self) -> tuple[str, ...]:
        return tuple(item.predicate_id for item in self.desired_predicates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TYPED_INTENT_SCHEMA,
            "intent_id": self.intent_id,
            "desired_predicates": [
                item.to_dict() for item in self.desired_predicates
            ],
            "source_refs": list(self.source_refs),
            "current_root_id": self.current_root_id,
            "metadata": _canonical(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypedIntent":
        predicates = (
            payload.get("desired_predicates")
            or payload.get("goals")
            or payload.get("predicates")
            or ()
        )
        return cls(
            intent_id=payload.get(
                "intent_id",
                payload.get("request_id", payload.get("goal_id", "")),
            ),
            desired_predicates=tuple(predicates),
            source_refs=tuple(
                payload.get("source_refs")
                or payload.get("provenance_refs")
                or ()
            ),
            current_root_id=payload.get(
                "current_root_id", payload.get("tree_id", "")
            ),
            metadata=payload.get("metadata") or {},
        )


DesiredBehavior = TypedIntent


@dataclass(frozen=True)
class ObservedFact:
    fact_id: str
    predicate: TypedPredicate
    truth: FactTruth
    authority: FactAuthority
    provenance_refs: tuple[str, ...]
    current_root_id: str = ""
    invalidation_selectors: tuple[InvalidationSelector, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _identifier(self.fact_id, "fact_id"))
        object.__setattr__(
            self, "predicate", _decode_record(self.predicate, TypedPredicate, "predicate")
        )
        object.__setattr__(self, "truth", _fact_truth(self.truth))
        object.__setattr__(
            self, "authority", _enum(self.authority, FactAuthority, "authority")
        )
        object.__setattr__(
            self,
            "provenance_refs",
            _ids(self.provenance_refs, "provenance_refs", required=True),
        )
        object.__setattr__(
            self,
            "current_root_id",
            _identifier(self.current_root_id, "current_root_id", required=False),
        )
        object.__setattr__(
            self,
            "invalidation_selectors",
            _decode_records(
                self.invalidation_selectors,
                InvalidationSelector,
                "invalidation_selectors",
                key="selector_id",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBSERVED_FACT_SCHEMA,
            "fact_id": self.fact_id,
            "predicate": self.predicate.to_dict(),
            "truth": self.truth.value,
            "authority": self.authority.value,
            "provenance_refs": list(self.provenance_refs),
            "current_root_id": self.current_root_id,
            "invalidation_selectors": [
                item.to_dict() for item in self.invalidation_selectors
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservedFact":
        raw_predicate = payload.get("predicate")
        if raw_predicate is None:
            raw_predicate = {
                key: payload[key]
                for key in (
                    "predicate_id",
                    "predicate_type",
                    "subject_ref",
                    "object_ref",
                    "polarity",
                    "support",
                    "property_id",
                )
                if key in payload
            }
        return cls(
            fact_id=payload.get("fact_id", payload.get("observation_id", "")),
            predicate=raw_predicate,
            truth=payload.get("truth", payload.get("value", FactTruth.UNKNOWN.value)),
            authority=payload.get(
                "authority", FactAuthority.BOUNDED_OBSERVATION.value
            ),
            provenance_refs=tuple(
                payload.get("provenance_refs")
                or payload.get("evidence_refs")
                or ()
            ),
            current_root_id=payload.get(
                "current_root_id", payload.get("tree_id", "")
            ),
            invalidation_selectors=tuple(
                payload.get("invalidation_selectors")
                or payload.get("invalidators")
                or ()
            ),
        )


CurrentFact = ObservedFact


@dataclass(frozen=True)
class ProducerRule:
    """A reviewed operator: all requirements are needed for every effect."""

    producer_id: str
    effect_predicate_ids: tuple[str, ...]
    required_predicate_ids: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    assumption_refs: tuple[str, ...] = ()
    invalidation_selectors: tuple[InvalidationSelector, ...] = ()
    proof_requirement_refs: tuple[str, ...] = ()
    validation_requirement_refs: tuple[str, ...] = ()
    task_candidate_ids: tuple[str, ...] = ()
    executable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id, "producer_id")
        )
        object.__setattr__(
            self,
            "effect_predicate_ids",
            _ids(
                self.effect_predicate_ids,
                "effect_predicate_ids",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "required_predicate_ids",
            _ids(self.required_predicate_ids, "required_predicate_ids"),
        )
        for name in (
            "provenance_refs",
            "assumption_refs",
            "proof_requirement_refs",
            "validation_requirement_refs",
            "task_candidate_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "invalidation_selectors",
            _decode_records(
                self.invalidation_selectors,
                InvalidationSelector,
                "invalidation_selectors",
                key="selector_id",
            ),
        )
        if not isinstance(self.executable, bool):
            raise ObligationCompilationError("executable must be a boolean")
        if not self.executable and self.task_candidate_ids:
            raise ObligationCompilationError(
                "non-executable logical producers cannot name task candidates"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRODUCER_RULE_SCHEMA,
            "producer_id": self.producer_id,
            "effect_predicate_ids": list(self.effect_predicate_ids),
            "required_predicate_ids": list(self.required_predicate_ids),
            "premise_mode": RefinementKind.AND.value,
            "provenance_refs": list(self.provenance_refs),
            "assumption_refs": list(self.assumption_refs),
            "invalidation_selectors": [
                item.to_dict() for item in self.invalidation_selectors
            ],
            "proof_requirement_refs": list(self.proof_requirement_refs),
            "validation_requirement_refs": list(self.validation_requirement_refs),
            "task_candidate_ids": list(self.task_candidate_ids),
            "executable": self.executable,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProducerRule":
        mode = str(payload.get("premise_mode", RefinementKind.AND.value)).lower()
        if mode != RefinementKind.AND.value:
            raise ObligationCompilationError(
                "producer premises must be AND requirements; alternatives are "
                "represented as separate producers"
            )
        return cls(
            producer_id=payload.get("producer_id", payload.get("id", "")),
            effect_predicate_ids=tuple(
                payload.get("effect_predicate_ids")
                or payload.get("produces")
                or payload.get("effects")
                or ()
            ),
            required_predicate_ids=tuple(
                payload.get("required_predicate_ids")
                or payload.get("requires")
                or payload.get("preconditions")
                or ()
            ),
            provenance_refs=tuple(
                payload.get("provenance_refs")
                or payload.get("source_refs")
                or ()
            ),
            assumption_refs=tuple(payload.get("assumption_refs") or ()),
            invalidation_selectors=tuple(
                payload.get("invalidation_selectors")
                or payload.get("invalidators")
                or ()
            ),
            proof_requirement_refs=tuple(
                payload.get("proof_requirement_refs")
                or payload.get("proof_requirements")
                or ()
            ),
            validation_requirement_refs=tuple(
                payload.get("validation_requirement_refs")
                or payload.get("validation_requirements")
                or ()
            ),
            task_candidate_ids=tuple(
                payload.get("task_candidate_ids")
                or payload.get("candidate_ids")
                or ()
            ),
            executable=payload.get(
                "executable", payload.get("requires_task", True)
            ),
        )


Producer = ProducerRule


@dataclass(frozen=True)
class TaskCandidate:
    candidate_id: str
    closes_obligation_ids: tuple[str, ...]
    producer_id: str = ""
    depends_on_candidate_ids: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _identifier(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self,
            "closes_obligation_ids",
            _ids(
                self.closes_obligation_ids,
                "closes_obligation_ids",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "producer_id",
            _identifier(self.producer_id, "producer_id", required=False),
        )
        object.__setattr__(
            self,
            "depends_on_candidate_ids",
            _ids(self.depends_on_candidate_ids, "depends_on_candidate_ids"),
        )
        object.__setattr__(
            self, "provenance_refs", _ids(self.provenance_refs, "provenance_refs")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "closes_obligation_ids": list(self.closes_obligation_ids),
            "producer_id": self.producer_id,
            "depends_on_candidate_ids": list(self.depends_on_candidate_ids),
            "provenance_refs": list(self.provenance_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskCandidate":
        return cls(
            candidate_id=payload.get(
                "candidate_id", payload.get("task_id", payload.get("id", ""))
            ),
            closes_obligation_ids=tuple(
                payload.get("closes_obligation_ids")
                or payload.get("closes")
                or payload.get("obligation_ids")
                or ()
            ),
            producer_id=payload.get("producer_id", ""),
            depends_on_candidate_ids=tuple(
                payload.get("depends_on_candidate_ids")
                or payload.get("depends_on")
                or ()
            ),
            provenance_refs=tuple(
                payload.get("provenance_refs")
                or payload.get("source_refs")
                or ()
            ),
        )


@dataclass(frozen=True)
class CompilationBounds:
    max_depth: int = 16
    max_nodes: int = 4_096
    max_producers: int = 1_024
    max_tasks: int = 4_096

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ObligationBoundsError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompilationBounds":
        return cls(
            **{
                name: payload[name]
                for name in cls.__dataclass_fields__
                if name in payload
            }
        )


def obligation_id_for_predicate(predicate_id: str) -> str:
    return f"obligation:predicate:{_identifier(predicate_id, 'predicate_id')}"


def obligation_id_for_producer(producer_id: str, effect_predicate_id: str) -> str:
    return (
        "obligation:producer:"
        + _identifier(producer_id, "producer_id")
        + ":for:"
        + _identifier(effect_predicate_id, "effect_predicate_id")
    )


@dataclass(frozen=True)
class ObligationNode:
    obligation_id: str
    kind: ObligationNodeKind
    status: ObligationStatus
    depth: int
    predicate_id: str = ""
    producer_id: str = ""
    parent_obligation_ids: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    assumption_refs: tuple[str, ...] = ()
    invalidation_selector_ids: tuple[str, ...] = ()
    proof_requirement_refs: tuple[str, ...] = ()
    validation_requirement_refs: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "obligation_id", _identifier(self.obligation_id, "obligation_id")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ObligationNodeKind, "kind")
        )
        object.__setattr__(
            self, "status", _enum(self.status, ObligationStatus, "status")
        )
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ObligationCompilationError("depth must be a non-negative integer")
        for name in ("predicate_id", "producer_id"):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name, required=False)
            )
        if self.kind is ObligationNodeKind.PRODUCER:
            if not self.producer_id or not self.predicate_id:
                raise ObligationCompilationError(
                    "producer obligations require producer_id and predicate_id"
                )
        elif not self.predicate_id:
            raise ObligationCompilationError(
                "goal/subgoal obligations require predicate_id"
            )
        for name in (
            "parent_obligation_ids",
            "provenance_refs",
            "assumption_refs",
            "invalidation_selector_ids",
            "proof_requirement_refs",
            "validation_requirement_refs",
            "reason_codes",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))

    @property
    def open(self) -> bool:
        return self.status is ObligationStatus.OPEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBLIGATION_NODE_SCHEMA,
            "obligation_id": self.obligation_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "depth": self.depth,
            "predicate_id": self.predicate_id,
            "producer_id": self.producer_id,
            "parent_obligation_ids": list(self.parent_obligation_ids),
            "provenance_refs": list(self.provenance_refs),
            "assumption_refs": list(self.assumption_refs),
            "invalidation_selector_ids": list(self.invalidation_selector_ids),
            "proof_requirement_refs": list(self.proof_requirement_refs),
            "validation_requirement_refs": list(
                self.validation_requirement_refs
            ),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObligationNode":
        return cls(
            obligation_id=payload.get("obligation_id", ""),
            kind=payload.get("kind", ""),
            status=payload.get("status", ""),
            depth=payload.get("depth", 0),
            predicate_id=payload.get("predicate_id", ""),
            producer_id=payload.get("producer_id", ""),
            parent_obligation_ids=tuple(
                payload.get("parent_obligation_ids") or ()
            ),
            provenance_refs=tuple(payload.get("provenance_refs") or ()),
            assumption_refs=tuple(payload.get("assumption_refs") or ()),
            invalidation_selector_ids=tuple(
                payload.get("invalidation_selector_ids") or ()
            ),
            proof_requirement_refs=tuple(
                payload.get("proof_requirement_refs") or ()
            ),
            validation_requirement_refs=tuple(
                payload.get("validation_requirement_refs") or ()
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )


Obligation = ObligationNode


@dataclass(frozen=True)
class ObligationRefinement:
    refinement_id: str
    parent_obligation_id: str
    kind: RefinementKind
    child_obligation_ids: tuple[str, ...]
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "refinement_id", _identifier(self.refinement_id, "refinement_id")
        )
        object.__setattr__(
            self,
            "parent_obligation_id",
            _identifier(self.parent_obligation_id, "parent_obligation_id"),
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, RefinementKind, "kind")
        )
        object.__setattr__(
            self,
            "child_obligation_ids",
            _ids(
                self.child_obligation_ids,
                "child_obligation_ids",
                required=True,
            ),
        )
        if self.parent_obligation_id in self.child_obligation_ids:
            raise ObligationCompilationError(
                "a refinement cannot directly contain its parent"
            )
        object.__setattr__(
            self, "provenance_refs", _ids(self.provenance_refs, "provenance_refs")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBLIGATION_REFINEMENT_SCHEMA,
            "refinement_id": self.refinement_id,
            "parent_obligation_id": self.parent_obligation_id,
            "kind": self.kind.value,
            "child_obligation_ids": list(self.child_obligation_ids),
            "provenance_refs": list(self.provenance_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObligationRefinement":
        return cls(
            refinement_id=payload.get("refinement_id", ""),
            parent_obligation_id=payload.get("parent_obligation_id", ""),
            kind=payload.get("kind", ""),
            child_obligation_ids=tuple(
                payload.get("child_obligation_ids") or ()
            ),
            provenance_refs=tuple(payload.get("provenance_refs") or ()),
        )


ObligationEdge = ObligationRefinement


@dataclass(frozen=True)
class ObligationIssue:
    issue_id: str
    kind: ObligationIssueKind
    severity: IssueSeverity
    obligation_ids: tuple[str, ...]
    predicate_ids: tuple[str, ...] = ()
    producer_ids: tuple[str, ...] = ()
    reason_code: str = ""
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue_id", _identifier(self.issue_id, "issue_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ObligationIssueKind, "kind")
        )
        object.__setattr__(
            self, "severity", _enum(self.severity, IssueSeverity, "severity")
        )
        for name in (
            "obligation_ids",
            "predicate_ids",
            "producer_ids",
            "provenance_refs",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "reason_code",
            _identifier(self.reason_code, "reason_code", required=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBLIGATION_ISSUE_SCHEMA,
            "issue_id": self.issue_id,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "obligation_ids": list(self.obligation_ids),
            "predicate_ids": list(self.predicate_ids),
            "producer_ids": list(self.producer_ids),
            "reason_code": self.reason_code,
            "provenance_refs": list(self.provenance_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObligationIssue":
        return cls(
            issue_id=payload.get("issue_id", ""),
            kind=payload.get("kind", ""),
            severity=payload.get("severity", ""),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            predicate_ids=tuple(payload.get("predicate_ids") or ()),
            producer_ids=tuple(payload.get("producer_ids") or ()),
            reason_code=payload.get("reason_code", ""),
            provenance_refs=tuple(payload.get("provenance_refs") or ()),
        )


GraphIssue = ObligationIssue


@dataclass(frozen=True)
class ObligationGraph:
    intent_id: str
    current_root_id: str
    predicates: tuple[TypedPredicate, ...]
    facts: tuple[ObservedFact, ...]
    producers: tuple[ProducerRule, ...]
    assumptions: tuple[AssumptionBinding, ...]
    task_candidates: tuple[TaskCandidate, ...]
    nodes: tuple[ObligationNode, ...]
    refinements: tuple[ObligationRefinement, ...]
    root_obligation_ids: tuple[str, ...]
    issues: tuple[ObligationIssue, ...]
    decision: ObligationGraphDecision
    bounds: CompilationBounds
    source_refs: tuple[str, ...] = ()
    interface: str = OBLIGATION_GRAPH_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _identifier(self.intent_id, "intent_id"))
        object.__setattr__(
            self,
            "current_root_id",
            _identifier(self.current_root_id, "current_root_id", required=False),
        )
        specifications = (
            ("predicates", TypedPredicate, "predicate_id"),
            ("facts", ObservedFact, "fact_id"),
            ("producers", ProducerRule, "producer_id"),
            ("assumptions", AssumptionBinding, "assumption_id"),
            ("task_candidates", TaskCandidate, "candidate_id"),
            ("nodes", ObligationNode, "obligation_id"),
            ("refinements", ObligationRefinement, "refinement_id"),
            ("issues", ObligationIssue, "issue_id"),
        )
        for name, cls, key in specifications:
            object.__setattr__(
                self,
                name,
                _decode_records(getattr(self, name), cls, name, key=key),
            )
        object.__setattr__(
            self,
            "root_obligation_ids",
            _ids(self.root_obligation_ids, "root_obligation_ids", required=True),
        )
        object.__setattr__(
            self, "decision", _enum(self.decision, ObligationGraphDecision, "decision")
        )
        if not isinstance(self.bounds, CompilationBounds):
            object.__setattr__(
                self, "bounds", CompilationBounds.from_dict(self.bounds)
            )
        object.__setattr__(
            self, "source_refs", _ids(self.source_refs, "source_refs", required=True)
        )
        object.__setattr__(
            self, "interface", _identifier(self.interface, "interface")
        )
        if self.interface != OBLIGATION_GRAPH_INTERFACE:
            raise ObligationCompilationError("unsupported obligation graph interface")
        self._validate_references()

    def _validate_references(self) -> None:
        predicate_ids = {item.predicate_id for item in self.predicates}
        producer_ids = {item.producer_id for item in self.producers}
        assumption_ids = {item.assumption_id for item in self.assumptions}
        node_ids = {item.obligation_id for item in self.nodes}
        candidate_ids = {item.candidate_id for item in self.task_candidates}
        if not set(self.root_obligation_ids) <= node_ids:
            raise ObligationCompilationError("graph references an unknown root")
        for node in self.nodes:
            if node.predicate_id not in predicate_ids:
                raise ObligationCompilationError(
                    f"node {node.obligation_id} references an unknown predicate"
                )
            if node.producer_id and node.producer_id not in producer_ids:
                raise ObligationCompilationError(
                    f"node {node.obligation_id} references an unknown producer"
                )
            if not set(node.parent_obligation_ids) <= node_ids:
                raise ObligationCompilationError(
                    f"node {node.obligation_id} has an unknown parent"
                )
            if not set(node.assumption_refs) <= assumption_ids:
                # Invalid assumption references are retained as graph issues by
                # the compiler but cannot be serialized as if they were bound.
                raise ObligationCompilationError(
                    f"node {node.obligation_id} has an unbound assumption"
                )
        for refinement in self.refinements:
            if refinement.parent_obligation_id not in node_ids or not set(
                refinement.child_obligation_ids
            ) <= node_ids:
                raise ObligationCompilationError(
                    f"refinement {refinement.refinement_id} has an unknown endpoint"
                )
        for candidate in self.task_candidates:
            if not set(candidate.depends_on_candidate_ids) <= candidate_ids:
                raise ObligationCompilationError(
                    f"candidate {candidate.candidate_id} has an unknown dependency"
                )

    @property
    def graph_id(self) -> str:
        return _content_id(
            "obligation-graph", self.to_dict(include_graph_id=False)
        )

    @property
    def ready(self) -> bool:
        return self.decision is ObligationGraphDecision.READY

    @property
    def planning_blocked(self) -> bool:
        return self.decision is ObligationGraphDecision.BLOCKED

    @property
    def review_required(self) -> bool:
        return self.decision is ObligationGraphDecision.REVIEW_REQUIRED

    @property
    def complete(self) -> bool:
        by_id = {item.obligation_id: item for item in self.nodes}
        return all(
            by_id[item].status is ObligationStatus.DISCHARGED
            for item in self.root_obligation_ids
        )

    @property
    def leaf_obligation_ids(self) -> tuple[str, ...]:
        # Producer obligations are executable leaves even when their AND
        # refinement records prerequisite dependencies.  Predicate
        # obligations are leaves only when no producer strategy refines them.
        predicate_strategy_parents = {
            item.parent_obligation_id
            for item in self.refinements
            if item.kind is RefinementKind.OR
        }
        executable_producers = {
            item.producer_id for item in self.producers if item.executable
        }
        return tuple(
            item.obligation_id
            for item in self.nodes
            if (
                item.producer_id in executable_producers
                if item.kind is ObligationNodeKind.PRODUCER
                else item.obligation_id not in predicate_strategy_parents
            )
        )

    @property
    def open_leaf_obligation_ids(self) -> tuple[str, ...]:
        leaves = set(self.leaf_obligation_ids)
        return tuple(
            item.obligation_id
            for item in self.nodes
            if item.obligation_id in leaves
            and item.status in {ObligationStatus.OPEN, ObligationStatus.REVIEW}
        )

    @property
    def uncovered_leaf_obligation_ids(self) -> tuple[str, ...]:
        covered = {
            obligation_id
            for candidate in self.task_candidates
            for obligation_id in candidate.closes_obligation_ids
        }
        return tuple(
            item for item in self.open_leaf_obligation_ids if item not in covered
        )

    def node(self, obligation_id: str) -> ObligationNode:
        wanted = _identifier(obligation_id, "obligation_id")
        for item in self.nodes:
            if item.obligation_id == wanted:
                return item
        raise KeyError(wanted)

    def children(self, obligation_id: str) -> tuple[ObligationNode, ...]:
        by_id = {item.obligation_id: item for item in self.nodes}
        child_ids: set[str] = set()
        for refinement in self.refinements:
            if refinement.parent_obligation_id == obligation_id:
                child_ids.update(refinement.child_obligation_ids)
        return tuple(by_id[item] for item in sorted(child_ids))

    def refinements_for(
        self, obligation_id: str
    ) -> tuple[ObligationRefinement, ...]:
        return tuple(
            item
            for item in self.refinements
            if item.parent_obligation_id == obligation_id
        )

    def issues_of_kind(
        self, kind: ObligationIssueKind | str
    ) -> tuple[ObligationIssue, ...]:
        normalized = _enum(kind, ObligationIssueKind, "kind")
        return tuple(item for item in self.issues if item.kind is normalized)

    def invalidated_by(self, changes: Mapping[str, Any]) -> tuple[str, ...]:
        selector_ids = {
            selector.selector_id
            for assumption in self.assumptions
            for selector in assumption.invalidation_selectors
            if selector.matches(changes)
        } | {
            selector.selector_id
            for predicate in self.predicates
            for selector in predicate.invalidation_selectors
            if selector.matches(changes)
        } | {
            selector.selector_id
            for producer in self.producers
            for selector in producer.invalidation_selectors
            if selector.matches(changes)
        }
        return tuple(
            item.obligation_id
            for item in self.nodes
            if selector_ids.intersection(item.invalidation_selector_ids)
        )

    def to_dict(self, *, include_graph_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": OBLIGATION_GRAPH_SCHEMA,
            "interface": self.interface,
            "intent_id": self.intent_id,
            "current_root_id": self.current_root_id,
            "predicates": [item.to_dict() for item in self.predicates],
            "facts": [item.to_dict() for item in self.facts],
            "producers": [item.to_dict() for item in self.producers],
            "assumptions": [item.to_dict() for item in self.assumptions],
            "task_candidates": [
                item.to_dict() for item in self.task_candidates
            ],
            "nodes": [item.to_dict() for item in self.nodes],
            "refinements": [item.to_dict() for item in self.refinements],
            "root_obligation_ids": list(self.root_obligation_ids),
            "issues": [item.to_dict() for item in self.issues],
            "decision": self.decision.value,
            "bounds": self.bounds.to_dict(),
            "source_refs": list(self.source_refs),
            "authority": {
                "proof_authority": False,
                "completion_authority": False,
                "candidate_generation_only": True,
                "unsupported_semantics_are_unknown": True,
            },
        }
        if include_graph_id:
            payload["graph_id"] = self.graph_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObligationGraph":
        if payload.get("schema") not in (None, OBLIGATION_GRAPH_SCHEMA):
            raise ObligationCompilationError("unsupported obligation graph schema")
        graph = cls(
            intent_id=payload.get("intent_id", ""),
            current_root_id=payload.get("current_root_id", ""),
            predicates=tuple(payload.get("predicates") or ()),
            facts=tuple(payload.get("facts") or ()),
            producers=tuple(payload.get("producers") or ()),
            assumptions=tuple(payload.get("assumptions") or ()),
            task_candidates=tuple(payload.get("task_candidates") or ()),
            nodes=tuple(payload.get("nodes") or ()),
            refinements=tuple(payload.get("refinements") or ()),
            root_obligation_ids=tuple(
                payload.get("root_obligation_ids") or ()
            ),
            issues=tuple(payload.get("issues") or ()),
            decision=payload.get("decision", ""),
            bounds=payload.get("bounds") or {},
            source_refs=tuple(payload.get("source_refs") or ()),
            interface=payload.get("interface", OBLIGATION_GRAPH_INTERFACE),
        )
        _verify_claimed_id(payload, graph.graph_id, "graph_id", "content_id")
        return graph

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> "ObligationGraph":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise ObligationCompilationError("obligation graph JSON must be an object")
        return cls.from_dict(payload)


def _record_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        return {
            name: getattr(value, name)
            for name in fields
            if hasattr(value, name)
        }
    return {}


def _formal_plan_projection(
    plan: Any,
) -> tuple[
    TypedIntent,
    tuple[TypedPredicate, ...],
    tuple[ProducerRule, ...],
    tuple[TaskCandidate, ...],
]:
    """Project ``FormalWorkPlan`` identifiers without interpreting formulas."""

    data = _record_mapping(plan)
    plan_id = str(
        getattr(plan, "content_id", "")
        or data.get("content_id")
        or _content_id("formal-work-plan-source", data)
    )
    goals = tuple(getattr(plan, "goals", data.get("goals", ())) or ())
    subgoals = tuple(
        getattr(plan, "subgoals", data.get("subgoals", ())) or ()
    )
    tasks = tuple(getattr(plan, "tasks", data.get("tasks", ())) or ())
    preconditions = {
        str(getattr(item, "precondition_id", _record_mapping(item).get("precondition_id", ""))):
        str(getattr(item, "formula_id", _record_mapping(item).get("formula_id", "")))
        for item in tuple(
            getattr(plan, "preconditions", data.get("preconditions", ())) or ()
        )
    }
    requirements = {
        str(getattr(item, "requirement_id", _record_mapping(item).get("requirement_id", "")))
        for item in tuple(
            getattr(
                plan,
                "evidence_requirements",
                data.get("evidence_requirements", ()),
            )
            or ()
        )
    }

    predicates: dict[str, TypedPredicate] = {}

    def formula_predicate(
        formula_id: str, source_ref: str, *, validation_refs: Iterable[str] = ()
    ) -> TypedPredicate:
        predicate_id = f"formal-formula:{formula_id}"
        if predicate_id not in predicates:
            predicates[predicate_id] = TypedPredicate(
                predicate_id=predicate_id,
                predicate_type="formal_formula_satisfied",
                subject_ref=formula_id,
                provenance_refs=(plan_id, source_ref),
                validation_requirement_refs=tuple(validation_refs),
            )
        return predicates[predicate_id]

    desired: list[TypedPredicate] = []
    goal_formula: dict[str, str] = {}
    for goal in goals:
        item = _record_mapping(goal)
        goal_id = str(getattr(goal, "goal_id", item.get("goal_id", "")))
        formula_id = str(
            getattr(
                goal,
                "satisfaction_formula_id",
                item.get("satisfaction_formula_id", ""),
            )
        )
        evidence_ids = tuple(
            getattr(
                goal,
                "evidence_requirement_ids",
                item.get("evidence_requirement_ids", ()),
            )
            or ()
        )
        evidence_ids = tuple(value for value in evidence_ids if value in requirements)
        predicate = formula_predicate(
            formula_id, f"formal-goal:{goal_id}", validation_refs=evidence_ids
        )
        desired.append(predicate)
        goal_formula[goal_id] = predicate.predicate_id

    producers: list[ProducerRule] = []
    subgoal_formula: dict[str, str] = {}
    for subgoal in subgoals:
        item = _record_mapping(subgoal)
        subgoal_id = str(
            getattr(subgoal, "subgoal_id", item.get("subgoal_id", ""))
        )
        goal_id = str(getattr(subgoal, "goal_id", item.get("goal_id", "")))
        formula_id = str(
            getattr(
                subgoal,
                "satisfaction_formula_id",
                item.get("satisfaction_formula_id", ""),
            )
        )
        predicate = formula_predicate(
            formula_id, f"formal-subgoal:{subgoal_id}"
        )
        subgoal_formula[subgoal_id] = predicate.predicate_id
        if goal_id in goal_formula:
            producers.append(
                ProducerRule(
                    producer_id=f"formal-refinement:{subgoal_id}",
                    effect_predicate_ids=(goal_formula[goal_id],),
                    required_predicate_ids=(predicate.predicate_id,),
                    provenance_refs=(plan_id, f"formal-subgoal:{subgoal_id}"),
                    executable=False,
                )
            )

    candidates: list[TaskCandidate] = []
    for task in tasks:
        item = _record_mapping(task)
        task_id = str(getattr(task, "task_id", item.get("task_id", "")))
        goal_id = str(getattr(task, "goal_id", item.get("goal_id", "")))
        subgoal_id = str(
            getattr(task, "subgoal_id", item.get("subgoal_id", ""))
        )
        effect_id = subgoal_formula.get(subgoal_id) or goal_formula.get(goal_id)
        if not effect_id:
            # A task with no typed goal/subgoal target cannot be promoted into
            # semantic proof; retain no invented relation.
            continue
        raw_precondition_ids = tuple(
            getattr(
                task, "precondition_ids", item.get("precondition_ids", ())
            )
            or ()
        )
        required_ids: list[str] = []
        for precondition_id in raw_precondition_ids:
            formula_id = preconditions.get(str(precondition_id))
            if formula_id:
                required_ids.append(
                    formula_predicate(
                        formula_id, f"formal-precondition:{precondition_id}"
                    ).predicate_id
                )
        evidence_ids = tuple(
            getattr(
                task,
                "evidence_requirement_ids",
                item.get("evidence_requirement_ids", ()),
            )
            or ()
        )
        producer_id = f"formal-task:{task_id}"
        producers.append(
            ProducerRule(
                producer_id=producer_id,
                effect_predicate_ids=(effect_id,),
                required_predicate_ids=tuple(required_ids),
                provenance_refs=(plan_id, f"formal-task:{task_id}"),
                validation_requirement_refs=tuple(
                    value for value in evidence_ids if value in requirements
                ),
                task_candidate_ids=(task_id,),
            )
        )
        candidates.append(
            TaskCandidate(
                candidate_id=task_id,
                closes_obligation_ids=(
                    obligation_id_for_producer(producer_id, effect_id),
                ),
                producer_id=producer_id,
                depends_on_candidate_ids=tuple(
                    getattr(
                        task,
                        "depends_on",
                        item.get("depends_on", ()),
                    )
                    or ()
                ),
                provenance_refs=(plan_id, f"formal-task:{task_id}"),
            )
        )

    tree_id = str(
        getattr(plan, "repository_tree_id", data.get("repository_tree_id", ""))
        or ""
    )
    intent = TypedIntent(
        intent_id=f"formal-plan-intent:{plan_id}",
        desired_predicates=tuple(desired),
        source_refs=(plan_id,),
        current_root_id=tree_id,
    )
    return intent, tuple(predicates.values()), tuple(producers), tuple(candidates)


class ObligationGraphCompiler:
    """Compile a finite AND/OR graph by deterministic backward chaining."""

    def __init__(
        self,
        *,
        bounds: CompilationBounds | Mapping[str, Any] | None = None,
        property_catalog: Any = None,
    ) -> None:
        if bounds is None:
            self.bounds = CompilationBounds()
        elif isinstance(bounds, CompilationBounds):
            self.bounds = bounds
        elif isinstance(bounds, Mapping):
            self.bounds = CompilationBounds.from_dict(bounds)
        else:
            raise ObligationCompilationError(
                "bounds must be CompilationBounds or a mapping"
            )
        self.property_catalog = property_catalog

    def compile(
        self,
        intent: TypedIntent | Mapping[str, Any] | Sequence[TypedPredicate] | None = None,
        current_facts: Sequence[ObservedFact | Mapping[str, Any]] = (),
        producers: Sequence[ProducerRule | Mapping[str, Any]] = (),
        *,
        typed_intent: TypedIntent | Mapping[str, Any] | None = None,
        facts: Sequence[ObservedFact | Mapping[str, Any]] | None = None,
        assumptions: Sequence[AssumptionBinding | Mapping[str, Any]] = (),
        task_candidates: Sequence[TaskCandidate | Mapping[str, Any]] = (),
        predicates: Sequence[TypedPredicate | Mapping[str, Any]] = (),
        formal_work_plan: Any = None,
        logic_goals: Sequence[Any] = (),
        code_proof_obligations: Sequence[Any] = (),
        query_plan: Any = None,
        evidence_bundle: Any = None,
        current_root_id: str = "",
    ) -> ObligationGraph:
        if typed_intent is not None:
            if intent is not None:
                raise ObligationCompilationError(
                    "supply intent or typed_intent, not both"
                )
            intent = typed_intent
        if facts is not None:
            if current_facts:
                raise ObligationCompilationError(
                    "supply current_facts or facts, not both"
                )
            current_facts = facts

        projected_predicates: tuple[TypedPredicate, ...] = ()
        projected_producers: tuple[ProducerRule, ...] = ()
        projected_candidates: tuple[TaskCandidate, ...] = ()
        if formal_work_plan is not None:
            (
                projected_intent,
                projected_predicates,
                projected_producers,
                projected_candidates,
            ) = _formal_plan_projection(formal_work_plan)
            if intent is None:
                intent = projected_intent

        if intent is None and logic_goals:
            logic_predicates: list[TypedPredicate] = []
            logic_sources: list[str] = []
            for goal in logic_goals:
                data = _record_mapping(goal)
                goal_id = str(
                    getattr(goal, "goal_id", data.get("goal_id", ""))
                )
                statement_ref = str(
                    getattr(
                        goal,
                        "positive_statement_ref",
                        data.get("positive_statement_ref", ""),
                    )
                )
                if not goal_id or not statement_ref:
                    raise ObligationCompilationError(
                        "logic goals require goal_id and positive_statement_ref"
                    )
                source_id = str(
                    getattr(goal, "content_id", "")
                    or data.get("content_id")
                    or f"logic-goal:{goal_id}"
                )
                logic_sources.append(source_id)
                logic_predicates.append(
                    TypedPredicate(
                        predicate_id=f"logic-statement:{statement_ref}",
                        predicate_type="program_logic_statement",
                        subject_ref=statement_ref,
                        support=(
                            SemanticSupport.UNSUPPORTED
                            if tuple(
                                getattr(
                                    goal,
                                    "unsupported_facets",
                                    data.get("unsupported_facets", ()),
                                )
                                or ()
                            )
                            else SemanticSupport.REVIEWED
                        ),
                        provenance_refs=(source_id,),
                    )
                )
            intent = TypedIntent(
                intent_id=_content_id(
                    "logic-goal-intent",
                    [item.to_dict() for item in logic_predicates],
                ),
                desired_predicates=tuple(logic_predicates),
                source_refs=tuple(logic_sources),
                current_root_id=current_root_id,
            )

        normalized_intent = self._intent(intent, current_root_id=current_root_id)
        root_id = _identifier(
            current_root_id or normalized_intent.current_root_id,
            "current_root_id",
            required=False,
        )
        normalized_facts = _decode_records(
            current_facts, ObservedFact, "current_facts", key="fact_id"
        )
        normalized_assumptions = _decode_records(
            assumptions,
            AssumptionBinding,
            "assumptions",
            key="assumption_id",
        )
        normalized_producers = _decode_records(
            tuple(producers) + projected_producers,
            ProducerRule,
            "producers",
            key="producer_id",
        )
        normalized_candidates = _decode_records(
            tuple(task_candidates) + projected_candidates,
            TaskCandidate,
            "task_candidates",
            key="candidate_id",
        )
        if len(normalized_producers) > self.bounds.max_producers:
            raise ObligationBoundsError("producer count exceeds max_producers")
        if len(normalized_candidates) > self.bounds.max_tasks:
            raise ObligationBoundsError("task candidate count exceeds max_tasks")
        if len(normalized_intent.desired_predicates) > self.bounds.max_nodes:
            raise ObligationBoundsError(
                "root obligation count exceeds max_nodes"
            )

        all_predicates: dict[str, TypedPredicate] = {}
        for predicate in (
            *normalized_intent.desired_predicates,
            *projected_predicates,
            *(
                _decode_record(item, TypedPredicate, "predicates")
                for item in predicates
            ),
        ):
            existing = all_predicates.get(predicate.predicate_id)
            if existing is not None and existing != predicate:
                raise ObligationCompilationError(
                    f"predicate_id {predicate.predicate_id!r} has conflicting definitions"
                )
            all_predicates[predicate.predicate_id] = predicate
        for fact in normalized_facts:
            existing = all_predicates.get(fact.predicate.predicate_id)
            if existing is None:
                all_predicates[fact.predicate.predicate_id] = fact.predicate
            elif existing.signed_key != fact.predicate.signed_key:
                raise ObligationCompilationError(
                    f"fact {fact.fact_id} conflicts with predicate identity"
                )

        normalized_facts, logic_sources = self._project_logic_goals(
            logic_goals, all_predicates, normalized_facts
        )
        self._bind_code_proof_obligations(
            code_proof_obligations, all_predicates
        )

        # All producer references must resolve before inference.  Missing
        # semantic atoms are not guessed from their identifier.
        for producer in normalized_producers:
            missing = set(
                producer.effect_predicate_ids + producer.required_predicate_ids
            ).difference(all_predicates)
            if missing:
                raise ObligationCompilationError(
                    f"producer {producer.producer_id} references unknown predicates: "
                    + ", ".join(sorted(missing))
                )

        issues: list[ObligationIssue] = []
        self._check_upstream_readiness(
            query_plan=query_plan,
            evidence_bundle=evidence_bundle,
            issues=issues,
        )
        self._check_catalog(all_predicates, issues)
        self._check_fact_set(normalized_facts, root_id, issues)
        self._check_desired_consistency(
            normalized_intent.desired_predicates, issues
        )

        assumption_by_id = {
            item.assumption_id: item for item in normalized_assumptions
        }
        producers_by_effect: dict[str, list[ProducerRule]] = {}
        for producer in normalized_producers:
            for effect_id in producer.effect_predicate_ids:
                producers_by_effect.setdefault(effect_id, []).append(producer)

        node_state: dict[str, dict[str, Any]] = {}
        refinements: dict[str, ObligationRefinement] = {}

        def add_issue(
            kind: ObligationIssueKind,
            severity: IssueSeverity,
            obligation_ids: Iterable[str],
            *,
            predicate_ids: Iterable[str] = (),
            producer_ids: Iterable[str] = (),
            reason_code: str,
            provenance_refs: Iterable[str] = (),
        ) -> None:
            payload = {
                "kind": kind.value,
                "severity": severity.value,
                "obligation_ids": sorted(set(obligation_ids)),
                "predicate_ids": sorted(set(predicate_ids)),
                "producer_ids": sorted(set(producer_ids)),
                "reason_code": reason_code,
                "provenance_refs": sorted(set(provenance_refs)),
            }
            issue_id = _content_id("obligation-issue", payload)
            if any(item.issue_id == issue_id for item in issues):
                return
            issues.append(
                ObligationIssue(
                    issue_id=issue_id,
                    kind=kind,
                    severity=severity,
                    obligation_ids=tuple(payload["obligation_ids"]),
                    predicate_ids=tuple(payload["predicate_ids"]),
                    producer_ids=tuple(payload["producer_ids"]),
                    reason_code=reason_code,
                    provenance_refs=tuple(payload["provenance_refs"]),
                )
            )

        def merge_node(
            obligation_id: str,
            *,
            kind: ObligationNodeKind,
            predicate: TypedPredicate,
            depth: int,
            parent_id: str = "",
            producer: ProducerRule | None = None,
            status: ObligationStatus = ObligationStatus.OPEN,
            reason_codes: Iterable[str] = (),
        ) -> bool:
            if obligation_id not in node_state and len(node_state) >= self.bounds.max_nodes:
                add_issue(
                    ObligationIssueKind.BOUND_EXCEEDED,
                    IssueSeverity.REVIEW,
                    (parent_id,) if parent_id else (),
                    predicate_ids=(predicate.predicate_id,),
                    reason_code="max_nodes_exceeded",
                    provenance_refs=predicate.provenance_refs,
                )
                return False
            assumptions_for_node = set(predicate.assumption_refs)
            selectors = {
                item.selector_id for item in predicate.invalidation_selectors
            }
            provenance = set(predicate.provenance_refs)
            proofs = set(predicate.proof_requirement_refs)
            validations = set(predicate.validation_requirement_refs)
            if producer is not None:
                assumptions_for_node.update(producer.assumption_refs)
                selectors.update(
                    item.selector_id for item in producer.invalidation_selectors
                )
                provenance.update(producer.provenance_refs)
                proofs.update(producer.proof_requirement_refs)
                validations.update(producer.validation_requirement_refs)
            for assumption_id in tuple(assumptions_for_node):
                assumption = assumption_by_id.get(assumption_id)
                if assumption is None:
                    assumptions_for_node.remove(assumption_id)
                    status = ObligationStatus.BLOCKED
                    add_issue(
                        ObligationIssueKind.INVALID_ASSUMPTION,
                        IssueSeverity.ERROR,
                        (obligation_id,),
                        predicate_ids=(predicate.predicate_id,),
                        producer_ids=(
                            (producer.producer_id,) if producer is not None else ()
                        ),
                        reason_code="unbound_assumption_ref",
                    )
                    continue
                selectors.update(
                    item.selector_id
                    for item in assumption.invalidation_selectors
                )
                provenance.update(assumption.provenance_refs)
                if assumption.status is AssumptionStatus.INVALID:
                    status = ObligationStatus.BLOCKED
                    add_issue(
                        ObligationIssueKind.INVALID_ASSUMPTION,
                        IssueSeverity.ERROR,
                        (obligation_id,),
                        predicate_ids=(predicate.predicate_id,),
                        reason_code="assumption_invalid",
                        provenance_refs=assumption.provenance_refs,
                    )
                elif assumption.status is AssumptionStatus.UNKNOWN:
                    if status is ObligationStatus.OPEN:
                        status = ObligationStatus.REVIEW
                    add_issue(
                        ObligationIssueKind.UNKNOWN_ASSUMPTION,
                        IssueSeverity.REVIEW,
                        (obligation_id,),
                        predicate_ids=(predicate.predicate_id,),
                        reason_code="assumption_status_unknown",
                        provenance_refs=assumption.provenance_refs,
                    )

            state = node_state.get(obligation_id)
            if state is None:
                node_state[obligation_id] = {
                    "kind": kind,
                    "predicate_id": predicate.predicate_id,
                    "producer_id": producer.producer_id if producer else "",
                    "status": status,
                    "depth": depth,
                    "parents": {parent_id} if parent_id else set(),
                    "provenance": provenance,
                    "assumptions": assumptions_for_node,
                    "selectors": selectors,
                    "proofs": proofs,
                    "validations": validations,
                    "reasons": set(reason_codes),
                }
                return True
            if parent_id:
                state["parents"].add(parent_id)
            state["depth"] = min(state["depth"], depth)
            state["provenance"].update(provenance)
            state["assumptions"].update(assumptions_for_node)
            state["selectors"].update(selectors)
            state["proofs"].update(proofs)
            state["validations"].update(validations)
            state["reasons"].update(reason_codes)
            precedence = {
                ObligationStatus.DISCHARGED: 0,
                ObligationStatus.OPEN: 1,
                ObligationStatus.REVIEW: 2,
                ObligationStatus.BLOCKED: 3,
                ObligationStatus.CONTRADICTED: 4,
            }
            if precedence[status] > precedence[state["status"]]:
                state["status"] = status
            return True

        facts_by_signed: dict[
            tuple[str, str, str, PredicatePolarity], list[ObservedFact]
        ] = {}
        for fact in normalized_facts:
            facts_by_signed.setdefault(fact.predicate.signed_key, []).append(fact)

        def observed_status(
            predicate: TypedPredicate,
        ) -> tuple[ObligationStatus, tuple[str, ...], tuple[str, ...]]:
            exact = facts_by_signed.get(predicate.signed_key, ())
            opposite = facts_by_signed.get(
                (*predicate.semantic_key, predicate.polarity.opposite), ()
            )
            exact_true = tuple(
                item
                for item in exact
                if item.truth is FactTruth.TRUE
                and item.authority.may_discharge
                and (not root_id or not item.current_root_id or item.current_root_id == root_id)
            )
            exact_false = tuple(
                item
                for item in exact
                if item.truth is FactTruth.FALSE
                and item.authority.may_discharge
                and (not root_id or not item.current_root_id or item.current_root_id == root_id)
            )
            opposite_true = tuple(
                item
                for item in opposite
                if item.truth is FactTruth.TRUE
                and item.authority.may_discharge
                and (not root_id or not item.current_root_id or item.current_root_id == root_id)
            )
            review = tuple(
                item
                for item in (*exact, *opposite)
                if item.truth in {FactTruth.UNKNOWN, FactTruth.REVIEW}
                or not item.authority.may_discharge
                or (root_id and item.current_root_id and item.current_root_id != root_id)
            )
            if exact_true and (exact_false or opposite_true):
                return (
                    ObligationStatus.CONTRADICTED,
                    tuple(
                        sorted(
                            {
                                ref
                                for item in (*exact_true, *exact_false, *opposite_true)
                                for ref in item.provenance_refs
                            }
                        )
                    ),
                    ("contradictory_current_facts",),
                )
            if exact_false or opposite_true:
                return (
                    ObligationStatus.CONTRADICTED,
                    tuple(
                        sorted(
                            {
                                ref
                                for item in (*exact_false, *opposite_true)
                                for ref in item.provenance_refs
                            }
                        )
                    ),
                    ("desired_predicate_refuted",),
                )
            if exact_true:
                return (
                    ObligationStatus.DISCHARGED,
                    tuple(
                        sorted(
                            {
                                ref
                                for item in exact_true
                                for ref in item.provenance_refs
                            }
                        )
                    ),
                    ("observed_fact_satisfies_predicate",),
                )
            if predicate.support is not SemanticSupport.REVIEWED or review:
                return (
                    ObligationStatus.REVIEW,
                    tuple(
                        sorted(
                            {
                                ref
                                for item in review
                                for ref in item.provenance_refs
                            }
                        )
                    ),
                    ("semantics_unknown_or_evidence_inconclusive",),
                )
            return ObligationStatus.OPEN, (), ()

        expanded: set[str] = set()

        def expand_predicate(
            predicate_id: str,
            depth: int,
            stack: tuple[str, ...],
            *,
            parent_id: str = "",
            root: bool = False,
        ) -> str:
            predicate = all_predicates[predicate_id]
            obligation_id = obligation_id_for_predicate(predicate_id)
            status, fact_provenance, reasons = observed_status(predicate)
            kind = ObligationNodeKind.GOAL if root else ObligationNodeKind.SUBGOAL
            if not merge_node(
                obligation_id,
                kind=kind,
                predicate=predicate,
                depth=depth,
                parent_id=parent_id,
                status=status,
                reason_codes=reasons,
            ):
                return ""
            node_state[obligation_id]["provenance"].update(fact_provenance)

            if predicate.support is not SemanticSupport.REVIEWED:
                add_issue(
                    ObligationIssueKind.UNSUPPORTED_SEMANTICS,
                    IssueSeverity.REVIEW,
                    (obligation_id,),
                    predicate_ids=(predicate_id,),
                    reason_code=(
                        "predicate_semantics_unsupported"
                        if predicate.support is SemanticSupport.UNSUPPORTED
                        else "predicate_semantics_unknown"
                    ),
                    provenance_refs=predicate.provenance_refs,
                )
                return obligation_id
            if status in {
                ObligationStatus.DISCHARGED,
                ObligationStatus.CONTRADICTED,
            }:
                if status is ObligationStatus.CONTRADICTED:
                    add_issue(
                        ObligationIssueKind.CONTRADICTION,
                        IssueSeverity.ERROR,
                        (obligation_id,),
                        predicate_ids=(predicate_id,),
                        reason_code=reasons[0],
                        provenance_refs=fact_provenance,
                    )
                return obligation_id
            if predicate_id in stack:
                cycle = stack[stack.index(predicate_id) :] + (predicate_id,)
                node_state[obligation_id]["status"] = ObligationStatus.BLOCKED
                node_state[obligation_id]["reasons"].add(
                    "backward_chaining_cycle"
                )
                add_issue(
                    ObligationIssueKind.CYCLE,
                    IssueSeverity.ERROR,
                    tuple(obligation_id_for_predicate(item) for item in cycle),
                    predicate_ids=cycle,
                    reason_code="backward_chaining_cycle",
                )
                return obligation_id
            if depth >= self.bounds.max_depth:
                node_state[obligation_id]["status"] = ObligationStatus.REVIEW
                node_state[obligation_id]["reasons"].add("max_depth_exceeded")
                add_issue(
                    ObligationIssueKind.BOUND_EXCEEDED,
                    IssueSeverity.REVIEW,
                    (obligation_id,),
                    predicate_ids=(predicate_id,),
                    reason_code="max_depth_exceeded",
                )
                return obligation_id
            if obligation_id in expanded:
                return obligation_id
            expanded.add(obligation_id)

            alternatives = tuple(
                sorted(
                    producers_by_effect.get(predicate_id, ()),
                    key=lambda item: item.producer_id,
                )
            )
            if not alternatives:
                return obligation_id

            producer_obligation_ids: list[str] = []
            for producer in alternatives:
                producer_obligation_id = obligation_id_for_producer(
                    producer.producer_id, predicate_id
                )
                if not merge_node(
                    producer_obligation_id,
                    kind=ObligationNodeKind.PRODUCER,
                    predicate=predicate,
                    depth=depth + 1,
                    parent_id=obligation_id,
                    producer=producer,
                ):
                    node_state[obligation_id]["status"] = ObligationStatus.REVIEW
                    node_state[obligation_id]["reasons"].add(
                        "max_nodes_exceeded"
                    )
                    continue
                producer_obligation_ids.append(producer_obligation_id)

                premise_predicates = [
                    all_predicates[item]
                    for item in producer.required_predicate_ids
                ]
                inconsistent: list[tuple[str, str]] = []
                for index, left in enumerate(premise_predicates):
                    for right in premise_predicates[index + 1 :]:
                        if left.is_opposite(right):
                            inconsistent.append(
                                (left.predicate_id, right.predicate_id)
                            )
                if inconsistent:
                    node_state[producer_obligation_id][
                        "status"
                    ] = ObligationStatus.BLOCKED
                    node_state[producer_obligation_id]["reasons"].add(
                        "producer_premises_inconsistent"
                    )
                    add_issue(
                        ObligationIssueKind.INCONSISTENT_PREMISE,
                        IssueSeverity.ERROR,
                        (producer_obligation_id,),
                        predicate_ids=tuple(
                            item for pair in inconsistent for item in pair
                        ),
                        producer_ids=(producer.producer_id,),
                        reason_code="producer_requires_opposite_predicates",
                        provenance_refs=producer.provenance_refs,
                    )

                child_ids: list[str] = []
                for required_id in producer.required_predicate_ids:
                    child_id = expand_predicate(
                        required_id,
                        depth + 2,
                        stack + (predicate_id,),
                        parent_id=producer_obligation_id,
                    )
                    if not child_id:
                        node_state[producer_obligation_id][
                            "status"
                        ] = ObligationStatus.REVIEW
                        node_state[producer_obligation_id]["reasons"].add(
                            "max_nodes_exceeded"
                        )
                        continue
                    child_ids.append(child_id)
                    child_status = node_state[child_ids[-1]]["status"]
                    if child_status in {
                        ObligationStatus.CONTRADICTED,
                        ObligationStatus.BLOCKED,
                    }:
                        node_state[producer_obligation_id][
                            "status"
                        ] = ObligationStatus.BLOCKED
                        node_state[producer_obligation_id]["reasons"].add(
                            "required_premise_blocked"
                        )
                        if child_status is ObligationStatus.CONTRADICTED:
                            add_issue(
                                ObligationIssueKind.INCONSISTENT_PREMISE,
                                IssueSeverity.ERROR,
                                (producer_obligation_id, child_ids[-1]),
                                predicate_ids=(required_id,),
                                producer_ids=(producer.producer_id,),
                                reason_code="required_premise_refuted_by_current_fact",
                                provenance_refs=producer.provenance_refs,
                            )
                    elif (
                        child_status is ObligationStatus.REVIEW
                        and node_state[producer_obligation_id]["status"]
                        is ObligationStatus.OPEN
                    ):
                        node_state[producer_obligation_id][
                            "status"
                        ] = ObligationStatus.REVIEW

                if child_ids:
                    refinement = ObligationRefinement(
                        refinement_id=_content_id(
                            "obligation-refinement",
                            {
                                "parent": producer_obligation_id,
                                "kind": RefinementKind.AND.value,
                                "children": sorted(child_ids),
                            },
                        ),
                        parent_obligation_id=producer_obligation_id,
                        kind=RefinementKind.AND,
                        child_obligation_ids=tuple(child_ids),
                        provenance_refs=producer.provenance_refs,
                    )
                    refinements[refinement.refinement_id] = refinement
                if not producer.executable:
                    child_statuses = {
                        node_state[item]["status"] for item in child_ids
                    }
                    if not producer.required_predicate_ids or (
                        child_statuses
                        and child_statuses == {ObligationStatus.DISCHARGED}
                    ):
                        node_state[producer_obligation_id][
                            "status"
                        ] = ObligationStatus.DISCHARGED
                        node_state[producer_obligation_id]["reasons"].add(
                            "reviewed_logical_producer_discharged"
                        )

            if not producer_obligation_ids:
                return obligation_id
            alternative_refinement = ObligationRefinement(
                refinement_id=_content_id(
                    "obligation-refinement",
                    {
                        "parent": obligation_id,
                        "kind": RefinementKind.OR.value,
                        "children": sorted(producer_obligation_ids),
                    },
                ),
                parent_obligation_id=obligation_id,
                kind=RefinementKind.OR,
                child_obligation_ids=tuple(producer_obligation_ids),
                provenance_refs=tuple(
                    sorted(
                        {
                            ref
                            for producer in alternatives
                            for ref in producer.provenance_refs
                        }
                    )
                ),
            )
            refinements[alternative_refinement.refinement_id] = alternative_refinement
            statuses = {
                node_state[item]["status"] for item in producer_obligation_ids
            }
            if ObligationStatus.DISCHARGED in statuses:
                node_state[obligation_id][
                    "status"
                ] = ObligationStatus.DISCHARGED
                node_state[obligation_id]["reasons"].add(
                    "producer_strategy_discharged"
                )
            elif statuses and statuses <= {
                ObligationStatus.BLOCKED,
                ObligationStatus.CONTRADICTED,
            }:
                node_state[obligation_id]["status"] = ObligationStatus.BLOCKED
                node_state[obligation_id]["reasons"].add(
                    "all_producer_strategies_blocked"
                )
            elif ObligationStatus.OPEN not in statuses and (
                ObligationStatus.REVIEW in statuses
            ):
                node_state[obligation_id]["status"] = ObligationStatus.REVIEW
            return obligation_id

        compiled_roots: list[str] = []
        for predicate in normalized_intent.desired_predicates:
            compiled_root = expand_predicate(
                predicate.predicate_id, 0, (), root=True
            )
            if compiled_root:
                compiled_roots.append(compiled_root)
        roots = tuple(compiled_roots)
        if len(roots) != len(normalized_intent.desired_predicates):
            raise ObligationBoundsError(
                "max_nodes cannot contain every root obligation"
            )

        # Producer declarations may name task candidates without requiring the
        # caller to duplicate the closure edge.
        candidate_by_id = {
            item.candidate_id: item for item in normalized_candidates
        }
        for producer in normalized_producers:
            for effect_id in producer.effect_predicate_ids:
                producer_node_id = obligation_id_for_producer(
                    producer.producer_id, effect_id
                )
                if producer_node_id not in node_state:
                    continue
                for candidate_id in producer.task_candidate_ids:
                    existing = candidate_by_id.get(candidate_id)
                    if existing is None:
                        candidate_by_id[candidate_id] = TaskCandidate(
                            candidate_id=candidate_id,
                            closes_obligation_ids=(producer_node_id,),
                            producer_id=producer.producer_id,
                            provenance_refs=producer.provenance_refs,
                        )

        predicate_strategy_parents = {
            refinement.parent_obligation_id
            for refinement in refinements.values()
            if refinement.kind is RefinementKind.OR
        }
        executable_producers = {
            item.producer_id
            for item in normalized_producers
            if item.executable
        }
        leaf_ids = {
            obligation_id
            for obligation_id, state in node_state.items()
            if (
                state["producer_id"] in executable_producers
                if state["kind"] is ObligationNodeKind.PRODUCER
                else obligation_id not in predicate_strategy_parents
            )
        }
        aliases: dict[str, str] = {}
        for node_id, state in node_state.items():
            aliases[node_id] = node_id
            if state["kind"] is ObligationNodeKind.PRODUCER:
                aliases.setdefault(state["producer_id"], node_id)
            else:
                aliases.setdefault(state["predicate_id"], node_id)

        closure_valid_candidates: list[TaskCandidate] = []
        for candidate in sorted(
            candidate_by_id.values(), key=lambda item: item.candidate_id
        ):
            closures: list[str] = []
            invalid: list[str] = []
            for reference in candidate.closes_obligation_ids:
                resolved = aliases.get(reference)
                if resolved is None or resolved not in leaf_ids:
                    invalid.append(reference)
                else:
                    closures.append(resolved)
            if invalid or not closures:
                add_issue(
                    ObligationIssueKind.INVALID_TASK_CLOSURE,
                    IssueSeverity.ERROR,
                    tuple(closures),
                    producer_ids=(
                        (candidate.producer_id,) if candidate.producer_id else ()
                    ),
                    reason_code=(
                        "task_candidate_closes_non_leaf_or_unknown_obligation"
                    ),
                    provenance_refs=candidate.provenance_refs,
                )
                continue
            closure_valid_candidates.append(
                replace(candidate, closes_obligation_ids=tuple(closures))
            )

        admitted_candidate_ids = {
            item.candidate_id for item in closure_valid_candidates
        }
        canonical_candidates: list[TaskCandidate] = []
        for candidate in closure_valid_candidates:
            missing_dependencies = set(
                candidate.depends_on_candidate_ids
            ).difference(admitted_candidate_ids)
            if missing_dependencies:
                add_issue(
                    ObligationIssueKind.INVALID_TASK_CLOSURE,
                    IssueSeverity.ERROR,
                    candidate.closes_obligation_ids,
                    producer_ids=(
                        (candidate.producer_id,) if candidate.producer_id else ()
                    ),
                    reason_code="task_candidate_dependency_not_admitted",
                    provenance_refs=candidate.provenance_refs,
                )
                continue
            canonical_candidates.append(candidate)
        valid_covered = {
            obligation_id
            for candidate in canonical_candidates
            for obligation_id in candidate.closes_obligation_ids
        }
        candidates_by_id = {
            item.candidate_id: item for item in canonical_candidates
        }
        visiting_candidates: list[str] = []
        visited_candidates: set[str] = set()

        def visit_candidate(candidate_id: str) -> None:
            if candidate_id in visited_candidates:
                return
            if candidate_id in visiting_candidates:
                cycle_ids = visiting_candidates[
                    visiting_candidates.index(candidate_id) :
                ] + [candidate_id]
                members = [
                    candidates_by_id[item]
                    for item in cycle_ids[:-1]
                ]
                add_issue(
                    ObligationIssueKind.CYCLE,
                    IssueSeverity.ERROR,
                    (
                        obligation_id
                        for candidate in members
                        for obligation_id in candidate.closes_obligation_ids
                    ),
                    producer_ids=(
                        candidate.producer_id
                        for candidate in members
                        if candidate.producer_id
                    ),
                    reason_code="task_candidate_dependency_cycle",
                    provenance_refs=(
                        ref
                        for candidate in members
                        for ref in candidate.provenance_refs
                    ),
                )
                return
            visiting_candidates.append(candidate_id)
            for dependency_id in candidates_by_id[
                candidate_id
            ].depends_on_candidate_ids:
                visit_candidate(dependency_id)
            visiting_candidates.pop()
            visited_candidates.add(candidate_id)

        for candidate_id in sorted(candidates_by_id):
            visit_candidate(candidate_id)

        for leaf_id in sorted(leaf_ids):
            state = node_state[leaf_id]
            if state["status"] not in {
                ObligationStatus.OPEN,
                ObligationStatus.REVIEW,
            }:
                continue
            if state["status"] is ObligationStatus.REVIEW:
                # Review/unknown is deliberately not transformed into a task.
                continue
            if leaf_id not in valid_covered:
                add_issue(
                    ObligationIssueKind.UNCOVERED_LEAF,
                    IssueSeverity.ERROR,
                    (leaf_id,),
                    predicate_ids=(state["predicate_id"],),
                    producer_ids=(
                        (state["producer_id"],) if state["producer_id"] else ()
                    ),
                    reason_code="open_leaf_has_no_task_candidate",
                    provenance_refs=state["provenance"],
                )

        nodes = tuple(
            ObligationNode(
                obligation_id=obligation_id,
                kind=state["kind"],
                status=state["status"],
                depth=state["depth"],
                predicate_id=state["predicate_id"],
                producer_id=state["producer_id"],
                parent_obligation_ids=tuple(state["parents"]),
                provenance_refs=tuple(state["provenance"]),
                assumption_refs=tuple(state["assumptions"]),
                invalidation_selector_ids=tuple(state["selectors"]),
                proof_requirement_refs=tuple(state["proofs"]),
                validation_requirement_refs=tuple(state["validations"]),
                reason_codes=tuple(state["reasons"]),
            )
            for obligation_id, state in sorted(node_state.items())
        )

        has_error = any(item.severity is IssueSeverity.ERROR for item in issues)
        has_review = any(item.severity is IssueSeverity.REVIEW for item in issues) or any(
            item.status is ObligationStatus.REVIEW for item in nodes
        )
        decision = (
            ObligationGraphDecision.BLOCKED
            if has_error
            else ObligationGraphDecision.REVIEW_REQUIRED
            if has_review
            else ObligationGraphDecision.READY
        )
        source_refs = set(normalized_intent.source_refs)
        source_refs.update(logic_sources)
        for item in normalized_facts:
            source_refs.update(item.provenance_refs)
        source_refs.update(
            str(getattr(query_plan, "plan_id", "")) for _ in (0,) if query_plan
        )
        source_refs.update(
            str(getattr(evidence_bundle, "bundle_id", ""))
            for _ in (0,)
            if evidence_bundle
        )
        source_refs.discard("")
        return ObligationGraph(
            intent_id=normalized_intent.intent_id,
            current_root_id=root_id,
            predicates=tuple(all_predicates.values()),
            facts=normalized_facts,
            producers=normalized_producers,
            assumptions=normalized_assumptions,
            task_candidates=tuple(canonical_candidates),
            nodes=nodes,
            refinements=tuple(refinements.values()),
            root_obligation_ids=roots,
            issues=tuple(issues),
            decision=decision,
            bounds=self.bounds,
            source_refs=tuple(source_refs),
        )

    def _intent(
        self,
        value: TypedIntent | Mapping[str, Any] | Sequence[TypedPredicate] | None,
        *,
        current_root_id: str,
    ) -> TypedIntent:
        if isinstance(value, TypedIntent):
            return value
        if isinstance(value, Mapping):
            return TypedIntent.from_dict(value)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            predicates = tuple(
                _decode_record(item, TypedPredicate, "intent")
                for item in value
            )
            identity = _content_id(
                "typed-intent",
                [item.to_dict() for item in predicates],
            )
            return TypedIntent(
                intent_id=identity,
                desired_predicates=predicates,
                source_refs=(identity,),
                current_root_id=current_root_id,
            )
        if value is None:
            raise ObligationCompilationError(
                "typed intent or formal_work_plan is required"
            )
        raise ObligationCompilationError(
            "intent must be TypedIntent, a typed mapping, or predicates"
        )

    def _check_catalog(
        self,
        predicates: Mapping[str, TypedPredicate],
        issues: list[ObligationIssue],
    ) -> None:
        if self.property_catalog is None:
            return
        get = getattr(self.property_catalog, "get", None)
        if not callable(get):
            raise ObligationCompilationError(
                "property_catalog must provide get(property_id)"
            )
        for predicate in predicates.values():
            if not predicate.property_id or get(predicate.property_id) is not None:
                continue
            issue_id = _content_id(
                "obligation-issue",
                {
                    "kind": ObligationIssueKind.UNSUPPORTED_SEMANTICS.value,
                    "predicate_id": predicate.predicate_id,
                    "property_id": predicate.property_id,
                },
            )
            issues.append(
                ObligationIssue(
                    issue_id=issue_id,
                    kind=ObligationIssueKind.UNSUPPORTED_SEMANTICS,
                    severity=IssueSeverity.REVIEW,
                    obligation_ids=(
                        obligation_id_for_predicate(predicate.predicate_id),
                    ),
                    predicate_ids=(predicate.predicate_id,),
                    reason_code="property_id_not_in_reviewed_catalog",
                    provenance_refs=predicate.provenance_refs,
                )
            )
            predicates[predicate.predicate_id] = replace(
                predicate, support=SemanticSupport.UNSUPPORTED
            )

    @staticmethod
    def _check_upstream_readiness(
        *,
        query_plan: Any,
        evidence_bundle: Any,
        issues: list[ObligationIssue],
    ) -> None:
        for value, kind, reason in (
            (
                query_plan,
                ObligationIssueKind.INCOMPLETE_QUERY_PLAN,
                "required_query_plan_not_ready",
            ),
            (
                evidence_bundle,
                ObligationIssueKind.INCOMPLETE_EVIDENCE,
                "required_evidence_bundle_not_ready",
            ),
        ):
            data = _record_mapping(value) if value is not None else {}
            ready = bool(getattr(value, "ready", False))
            if not ready and data:
                ready = str(data.get("decision", "")).casefold() == "ready"
            if value is None or ready:
                continue
            source_id = str(
                getattr(value, "plan_id", "")
                or getattr(value, "bundle_id", "")
                or getattr(value, "content_id", "")
            )
            issues.append(
                ObligationIssue(
                    issue_id=_content_id(
                        "obligation-issue",
                        {"kind": kind.value, "source": source_id, "reason": reason},
                    ),
                    kind=kind,
                    severity=IssueSeverity.ERROR,
                    obligation_ids=(),
                    reason_code=reason,
                    provenance_refs=(source_id,) if source_id else (),
                )
            )

    @staticmethod
    def _check_fact_set(
        facts: tuple[ObservedFact, ...],
        current_root_id: str,
        issues: list[ObligationIssue],
    ) -> None:
        true_by_semantic: dict[
            tuple[str, str, str], dict[PredicatePolarity, list[ObservedFact]]
        ] = {}
        for fact in facts:
            if (
                current_root_id
                and fact.current_root_id
                and fact.current_root_id != current_root_id
            ):
                issues.append(
                    ObligationIssue(
                        issue_id=_content_id(
                            "obligation-issue",
                            {
                                "kind": ObligationIssueKind.STALE_FACT.value,
                                "fact": fact.fact_id,
                            },
                        ),
                        kind=ObligationIssueKind.STALE_FACT,
                        severity=IssueSeverity.REVIEW,
                        obligation_ids=(),
                        predicate_ids=(fact.predicate.predicate_id,),
                        reason_code="fact_bound_to_different_root",
                        provenance_refs=fact.provenance_refs,
                    )
                )
            if not fact.authority.may_discharge and fact.truth in {
                FactTruth.TRUE,
                FactTruth.FALSE,
            }:
                issues.append(
                    ObligationIssue(
                        issue_id=_content_id(
                            "obligation-issue",
                            {
                                "kind": ObligationIssueKind.NON_AUTHORITATIVE_FACT.value,
                                "fact": fact.fact_id,
                            },
                        ),
                        kind=ObligationIssueKind.NON_AUTHORITATIVE_FACT,
                        severity=IssueSeverity.REVIEW,
                        obligation_ids=(),
                        predicate_ids=(fact.predicate.predicate_id,),
                        reason_code="nomination_or_diagnostic_fact_cannot_discharge",
                        provenance_refs=fact.provenance_refs,
                    )
                )
            current = (
                not current_root_id
                or not fact.current_root_id
                or fact.current_root_id == current_root_id
            )
            if (
                fact.truth in {FactTruth.TRUE, FactTruth.FALSE}
                and fact.authority.may_discharge
                and current
            ):
                asserted_polarity = (
                    fact.predicate.polarity
                    if fact.truth is FactTruth.TRUE
                    else fact.predicate.polarity.opposite
                )
                true_by_semantic.setdefault(
                    fact.predicate.semantic_key, {}
                ).setdefault(asserted_polarity, []).append(fact)
        for values in true_by_semantic.values():
            if set(values) != {
                PredicatePolarity.POSITIVE,
                PredicatePolarity.NEGATIVE,
            }:
                continue
            contradictory = (
                values[PredicatePolarity.POSITIVE]
                + values[PredicatePolarity.NEGATIVE]
            )
            issues.append(
                ObligationIssue(
                    issue_id=_content_id(
                        "obligation-issue",
                        {
                            "kind": ObligationIssueKind.CONTRADICTION.value,
                            "facts": sorted(item.fact_id for item in contradictory),
                        },
                    ),
                    kind=ObligationIssueKind.CONTRADICTION,
                    severity=IssueSeverity.ERROR,
                    obligation_ids=(),
                    predicate_ids=tuple(
                        item.predicate.predicate_id for item in contradictory
                    ),
                    reason_code="current_facts_assert_opposite_predicates",
                    provenance_refs=tuple(
                        ref
                        for item in contradictory
                        for ref in item.provenance_refs
                    ),
                )
            )

    @staticmethod
    def _check_desired_consistency(
        predicates: tuple[TypedPredicate, ...],
        issues: list[ObligationIssue],
    ) -> None:
        for index, left in enumerate(predicates):
            for right in predicates[index + 1 :]:
                if not left.is_opposite(right):
                    continue
                issues.append(
                    ObligationIssue(
                        issue_id=_content_id(
                            "obligation-issue",
                            {
                                "kind": ObligationIssueKind.CONTRADICTION.value,
                                "predicates": sorted(
                                    (left.predicate_id, right.predicate_id)
                                ),
                            },
                        ),
                        kind=ObligationIssueKind.CONTRADICTION,
                        severity=IssueSeverity.ERROR,
                        obligation_ids=(
                            obligation_id_for_predicate(left.predicate_id),
                            obligation_id_for_predicate(right.predicate_id),
                        ),
                        predicate_ids=(left.predicate_id, right.predicate_id),
                        reason_code="typed_intent_requires_opposite_predicates",
                        provenance_refs=left.provenance_refs
                        + right.provenance_refs,
                    )
                )

    @staticmethod
    def _project_logic_goals(
        logic_goals: Sequence[Any],
        predicates: dict[str, TypedPredicate],
        facts: tuple[ObservedFact, ...],
    ) -> tuple[tuple[ObservedFact, ...], tuple[str, ...]]:
        result = list(facts)
        sources: set[str] = set()
        for goal in logic_goals:
            data = _record_mapping(goal)
            goal_id = str(getattr(goal, "goal_id", data.get("goal_id", "")))
            statement_ref = str(
                getattr(
                    goal,
                    "positive_statement_ref",
                    data.get("positive_statement_ref", ""),
                )
            )
            if not goal_id or not statement_ref:
                raise ObligationCompilationError(
                    "logic goals require goal_id and positive_statement_ref"
                )
            source_id = str(
                getattr(goal, "content_id", "")
                or data.get("content_id")
                or f"logic-goal:{goal_id}"
            )
            sources.add(source_id)
            unsupported = tuple(
                getattr(
                    goal,
                    "unsupported_facets",
                    data.get("unsupported_facets", ()),
                )
                or ()
            )
            predicate_id = f"logic-statement:{statement_ref}"
            predicate = predicates.get(predicate_id)
            if predicate is None:
                predicate = TypedPredicate(
                    predicate_id=predicate_id,
                    predicate_type="program_logic_statement",
                    subject_ref=statement_ref,
                    support=(
                        SemanticSupport.UNSUPPORTED
                        if unsupported
                        else SemanticSupport.REVIEWED
                    ),
                    provenance_refs=(source_id,),
                    assumption_refs=tuple(
                        getattr(
                            goal,
                            "assumption_refs",
                            data.get("assumption_refs", ()),
                        )
                        or ()
                    ),
                    proof_requirement_refs=tuple(
                        getattr(
                            goal,
                            "translation_requirement_refs",
                            data.get("translation_requirement_refs", ()),
                        )
                        or ()
                    ),
                )
                predicates[predicate_id] = predicate
            disposition = str(
                getattr(
                    getattr(goal, "disposition", ""),
                    "value",
                    data.get("disposition", ""),
                )
            )
            proof_status = str(
                getattr(
                    getattr(goal, "proof_status", ""),
                    "value",
                    data.get("proof_status", ""),
                )
            )
            if disposition == "discharged" and proof_status in {
                "kernel_verified",
                "validated_refuted",
            }:
                result.append(
                    ObservedFact(
                        fact_id=f"logic-goal-fact:{goal_id}",
                        predicate=predicate,
                        truth=FactTruth.TRUE,
                        authority=FactAuthority.PROOF_RECEIPT,
                        provenance_refs=(source_id,),
                    )
                )
        return (
            tuple(sorted(result, key=lambda item: item.fact_id)),
            tuple(sorted(sources)),
        )

    @staticmethod
    def _bind_code_proof_obligations(
        obligations: Sequence[Any],
        predicates: dict[str, TypedPredicate],
    ) -> None:
        by_goal: dict[str, list[str]] = {}
        for obligation in obligations:
            data = _record_mapping(obligation)
            goal_id = str(
                getattr(obligation, "goal_id", data.get("goal_id", ""))
            )
            obligation_id = str(
                getattr(
                    obligation,
                    "obligation_id",
                    data.get("obligation_id", data.get("content_id", "")),
                )
            )
            if not goal_id or not obligation_id:
                raise ObligationCompilationError(
                    "code proof obligations require goal_id and obligation_id"
                )
            by_goal.setdefault(goal_id, []).append(obligation_id)
        for predicate_id, predicate in tuple(predicates.items()):
            refs = by_goal.get(predicate_id) or by_goal.get(predicate.subject_ref)
            if refs:
                predicates[predicate_id] = replace(
                    predicate,
                    proof_requirement_refs=tuple(
                        sorted(set(predicate.proof_requirement_refs).union(refs))
                    ),
                )


def compile_obligation_graph(
    intent: TypedIntent | Mapping[str, Any] | Sequence[TypedPredicate] | None = None,
    current_facts: Sequence[ObservedFact | Mapping[str, Any]] = (),
    producers: Sequence[ProducerRule | Mapping[str, Any]] = (),
    **kwargs: Any,
) -> ObligationGraph:
    """Functional entry point for :class:`ObligationGraphCompiler`."""

    bounds = kwargs.pop("bounds", None)
    property_catalog = kwargs.pop("property_catalog", None)
    return ObligationGraphCompiler(
        bounds=bounds, property_catalog=property_catalog
    ).compile(intent, current_facts, producers, **kwargs)


compile_desired_observed_obligation_graph = compile_obligation_graph
compile_and_or_obligation_graph = compile_obligation_graph
BackwardChainingObligationCompiler = ObligationGraphCompiler
ANDORObligationGraphCompiler = ObligationGraphCompiler


__all__ = [
    "ASSUMPTION_SCHEMA",
    "INVALIDATION_SELECTOR_SCHEMA",
    "OBLIGATION_GRAPH_INTERFACE",
    "OBLIGATION_GRAPH_SCHEMA",
    "OBSERVED_FACT_SCHEMA",
    "PRODUCER_RULE_SCHEMA",
    "TASK_CANDIDATE_SCHEMA",
    "TYPED_INTENT_SCHEMA",
    "TYPED_PREDICATE_SCHEMA",
    "ANDORObligationGraphCompiler",
    "AssumptionBinding",
    "AssumptionStatus",
    "BackwardChainingObligationCompiler",
    "CompilationBounds",
    "CurrentFact",
    "DesiredBehavior",
    "FactAuthority",
    "FactTruth",
    "GraphIssue",
    "InvalidationSelector",
    "InvalidationSelectorKind",
    "IssueSeverity",
    "Obligation",
    "ObligationBoundsError",
    "ObligationCompilationError",
    "ObligationEdge",
    "ObligationGraph",
    "ObligationGraphCompiler",
    "ObligationGraphDecision",
    "ObligationIssue",
    "ObligationIssueKind",
    "ObligationNode",
    "ObligationNodeKind",
    "ObligationRefinement",
    "ObligationStatus",
    "Predicate",
    "PredicatePolarity",
    "Producer",
    "ProducerRule",
    "RefinementKind",
    "SemanticSupport",
    "TaskCandidate",
    "TypedIntent",
    "TypedPredicate",
    "compile_and_or_obligation_graph",
    "compile_desired_observed_obligation_graph",
    "compile_obligation_graph",
    "obligation_id_for_predicate",
    "obligation_id_for_producer",
]
