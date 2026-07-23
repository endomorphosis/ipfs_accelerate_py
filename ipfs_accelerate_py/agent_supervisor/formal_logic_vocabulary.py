"""Reviewed logic vocabulary for canonical supervisor work plans.

This module is intentionally a construction API, not a natural-language
parser.  Formulas can only use the predicates, operators, arities, and term
sorts reviewed here.  Model output may select an already reviewed template by
identifier, but cannot introduce predicates or turn prose into a formula.

The finite-trace evaluator and frame projection are bounded reference
semantics.  They provide plan-check and context-navigation artifacts only;
neither graph reachability nor a successful evaluation is a code proof.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
    _enum,
    _mapping,
    _text,
)


LOGIC_VOCABULARY_VERSION = 1
DCEC_VOCABULARY_VERSION = 1
TDFOL_VOCABULARY_VERSION = 1
FRAME_LOGIC_PROJECTION_VERSION = 1

FORMULA_SCHEMA = "ipfs_accelerate_py/agent-supervisor/reviewed-formula@1"
LOGIC_TERM_SCHEMA = "ipfs_accelerate_py/agent-supervisor/logic-term@1"
TRACE_FACT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/trace-fact@1"
TRACE_STEP_SCHEMA = "ipfs_accelerate_py/agent-supervisor/trace-step@1"
FINITE_TRACE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/finite-trace@1"
FRAME_PROJECTION_CONFIG_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/frame-projection-config@1"
)
FRAME_LOGIC_PROJECTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/frame-logic-projection@1"
)

LogicVocabularyValidationError = ContractValidationError


class TermSort(str, Enum):
    ACTOR = "actor"
    GOAL = "goal"
    SUBGOAL = "subgoal"
    TASK = "task"
    EVENT = "event"
    FLUENT = "fluent"
    EVIDENCE = "evidence"
    SYMBOL = "symbol"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    TIME = "time"


Sort = TermSort


class TermKind(str, Enum):
    CONSTANT = "constant"
    VARIABLE = "variable"


class ReviewedPredicate(str, Enum):
    TASK_READY = "task_ready"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_TERMINAL = "task_terminal"
    EVENT_OCCURRED = "event_occurred"
    DEPENDENCY_SATISFIED = "dependency_satisfied"
    DEADLINE_MET = "deadline_met"
    GOAL_SATISFIED = "goal_satisfied"
    EVIDENCE_AVAILABLE = "evidence_available"
    AUTHORIZED = "authorized"
    SAFE_STATE = "safe_state"


class FormulaOperator(str, Enum):
    ATOM = "atom"
    NOT = "not"
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    IFF = "iff"
    BELIEF = "belief"
    KNOWLEDGE = "knowledge"
    INTENTION = "intention"
    OBLIGATION = "obligation"
    PERMISSION = "permission"
    PROHIBITION = "prohibition"
    DELEGATION = "delegation"
    EXECUTION_EVENT = "execution_event"
    DEPENDENCY_ORDER = "dependency_order"
    DEADLINE = "deadline"
    LIVENESS = "liveness"
    SAFETY = "safety"
    GOAL_SATISFACTION = "goal_satisfaction"


class DCECOperator(str, Enum):
    BELIEF = "belief"
    KNOWLEDGE = "knowledge"
    INTENTION = "intention"
    OBLIGATION = "obligation"
    PERMISSION = "permission"
    PROHIBITION = "prohibition"
    DELEGATION = "delegation"
    EXECUTION_EVENT = "execution_event"


class TDFOLProperty(str, Enum):
    DEPENDENCY_ORDER = "dependency_order"
    DEADLINE = "deadline"
    LIVENESS = "liveness"
    SAFETY = "safety"
    GOAL_SATISFACTION = "goal_satisfaction"


_PREDICATE_SIGNATURES: Dict[ReviewedPredicate, Tuple[TermSort, ...]] = {
    ReviewedPredicate.TASK_READY: (TermSort.TASK,),
    ReviewedPredicate.TASK_STARTED: (TermSort.TASK,),
    ReviewedPredicate.TASK_COMPLETED: (TermSort.TASK,),
    ReviewedPredicate.TASK_TERMINAL: (TermSort.TASK,),
    ReviewedPredicate.EVENT_OCCURRED: (TermSort.EVENT,),
    ReviewedPredicate.DEPENDENCY_SATISFIED: (TermSort.TASK, TermSort.TASK),
    ReviewedPredicate.DEADLINE_MET: (TermSort.TASK, TermSort.TIME),
    ReviewedPredicate.GOAL_SATISFIED: (TermSort.GOAL,),
    ReviewedPredicate.EVIDENCE_AVAILABLE: (TermSort.EVIDENCE,),
    ReviewedPredicate.AUTHORIZED: (TermSort.ACTOR, TermSort.TASK),
    ReviewedPredicate.SAFE_STATE: (TermSort.SYMBOL,),
}


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ContractValidationError(
            "unsupported schema %r; expected %s" % (supplied, expected)
        )


def _positive(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ContractValidationError(
            "%s must be a %s integer" % (field_name, qualifier)
        )
    return value


@dataclass(frozen=True)
class LogicTerm(CanonicalContract):
    SCHEMA: ClassVar[str] = LOGIC_TERM_SCHEMA

    sort: TermSort
    value: Any
    kind: TermKind = TermKind.CONSTANT

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort", _enum(self.sort, TermSort, field_name="sort"))
        object.__setattr__(self, "kind", _enum(self.kind, TermKind, field_name="kind"))
        value = _canonical_value(self.value)
        if self.kind is TermKind.VARIABLE:
            value = _text(value, field_name="value", required=True)
        elif self.sort in (TermSort.INTEGER, TermSort.TIME):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractValidationError(
                    "%s terms require a non-negative integer" % self.sort.value
                )
        elif self.sort is TermSort.BOOLEAN:
            if not isinstance(value, bool):
                raise ContractValidationError("boolean terms require a boolean value")
        elif not isinstance(value, str) or not value.strip():
            raise ContractValidationError(
                "%s terms require a non-empty string" % self.sort.value
            )
        object.__setattr__(self, "value", value)

    def _payload(self) -> Dict[str, Any]:
        return {
            "vocabulary_version": LOGIC_VOCABULARY_VERSION,
            "sort": self.sort,
            "kind": self.kind,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicTerm":
        _schema(payload, cls.SCHEMA)
        return cls(
            sort=payload.get("sort", TermSort.SYMBOL),
            kind=payload.get("kind", TermKind.CONSTANT),
            value=payload.get("value"),
        )


Term = LogicTerm


def constant(sort: TermSort, value: Any) -> LogicTerm:
    return LogicTerm(sort=sort, value=value)


def variable(sort: TermSort, name: str) -> LogicTerm:
    return LogicTerm(sort=sort, value=name, kind=TermKind.VARIABLE)


def _term(value: Any) -> LogicTerm:
    if isinstance(value, LogicTerm):
        return value
    if isinstance(value, Mapping):
        return LogicTerm.from_dict(value)
    raise ContractValidationError("formula terms must be LogicTerm values")


@dataclass(frozen=True)
class Formula(CanonicalContract):
    """Closed formula AST accepted by the reviewed vocabulary."""

    SCHEMA: ClassVar[str] = FORMULA_SCHEMA

    operator: FormulaOperator
    predicate: Optional[ReviewedPredicate] = None
    terms: Tuple[LogicTerm, ...] = ()
    operands: Tuple["Formula", ...] = ()
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    profile_id: str = "supervisor-reviewed"
    profile_version: int = LOGIC_VOCABULARY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operator",
            _enum(self.operator, FormulaOperator, field_name="operator"),
        )
        object.__setattr__(
            self,
            "profile_id",
            _text(self.profile_id, field_name="profile_id", required=True),
        )
        object.__setattr__(
            self, "profile_version", _positive(self.profile_version, "profile_version")
        )
        terms = tuple(_term(item) for item in self.terms)
        operands: List[Formula] = []
        for item in self.operands:
            if isinstance(item, Formula):
                operands.append(item)
            elif isinstance(item, Mapping):
                operands.append(Formula.from_dict(item))
            else:
                raise ContractValidationError("formula operands must be Formula values")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "operands", tuple(operands))
        if self.lower_bound is not None:
            object.__setattr__(
                self,
                "lower_bound",
                _positive(self.lower_bound, "lower_bound", allow_zero=True),
            )
        if self.upper_bound is not None:
            object.__setattr__(
                self,
                "upper_bound",
                _positive(self.upper_bound, "upper_bound", allow_zero=True),
            )
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ContractValidationError(
                "formula lower_bound must not exceed upper_bound"
            )
        self._validate_shape()

    def _validate_shape(self) -> None:
        operator = self.operator
        if operator is FormulaOperator.ATOM:
            if self.predicate is None:
                raise ContractValidationError(
                    "atom formulas require a reviewed predicate"
                )
            predicate = _enum(self.predicate, ReviewedPredicate, field_name="predicate")
            object.__setattr__(self, "predicate", predicate)
            signature = _PREDICATE_SIGNATURES[predicate]
            actual = tuple(item.sort for item in self.terms)
            if actual != signature:
                raise ContractValidationError(
                    "predicate %s expects term sorts %s"
                    % (predicate.value, ", ".join(item.value for item in signature))
                )
            if self.operands:
                raise ContractValidationError("atom formulas cannot contain operands")
            return
        if self.predicate is not None:
            raise ContractValidationError("only atom formulas may name a predicate")

        unary = {
            FormulaOperator.NOT,
            FormulaOperator.BELIEF,
            FormulaOperator.KNOWLEDGE,
            FormulaOperator.INTENTION,
            FormulaOperator.OBLIGATION,
            FormulaOperator.PERMISSION,
            FormulaOperator.PROHIBITION,
            FormulaOperator.LIVENESS,
            FormulaOperator.SAFETY,
            FormulaOperator.GOAL_SATISFACTION,
        }
        binary = {FormulaOperator.IMPLIES, FormulaOperator.IFF}
        if operator in unary and len(self.operands) != 1:
            raise ContractValidationError("%s expects one operand" % operator.value)
        if operator in binary and len(self.operands) != 2:
            raise ContractValidationError("%s expects two operands" % operator.value)
        if (
            operator in (FormulaOperator.AND, FormulaOperator.OR)
            and len(self.operands) < 2
        ):
            raise ContractValidationError(
                "%s expects at least two operands" % operator.value
            )

        signatures: Dict[FormulaOperator, Tuple[TermSort, ...]] = {
            FormulaOperator.BELIEF: (TermSort.ACTOR, TermSort.TIME),
            FormulaOperator.KNOWLEDGE: (TermSort.ACTOR, TermSort.TIME),
            FormulaOperator.INTENTION: (TermSort.ACTOR, TermSort.TIME),
            FormulaOperator.OBLIGATION: (TermSort.ACTOR, TermSort.TIME),
            FormulaOperator.PERMISSION: (TermSort.ACTOR, TermSort.TIME),
            FormulaOperator.PROHIBITION: (TermSort.ACTOR, TermSort.TIME),
            FormulaOperator.DELEGATION: (
                TermSort.ACTOR,
                TermSort.ACTOR,
                TermSort.TASK,
                TermSort.TIME,
            ),
            FormulaOperator.EXECUTION_EVENT: (
                TermSort.ACTOR,
                TermSort.TASK,
                TermSort.EVENT,
                TermSort.TIME,
            ),
            FormulaOperator.DEPENDENCY_ORDER: (TermSort.TASK, TermSort.TASK),
            FormulaOperator.DEADLINE: (TermSort.TASK, TermSort.TIME),
        }
        if operator in signatures:
            actual = tuple(item.sort for item in self.terms)
            if actual != signatures[operator]:
                raise ContractValidationError(
                    "%s expects term sorts %s"
                    % (
                        operator.value,
                        ", ".join(item.value for item in signatures[operator]),
                    )
                )
        elif self.terms:
            raise ContractValidationError(
                "%s does not accept direct terms" % operator.value
            )
        if (
            operator
            in (
                FormulaOperator.DELEGATION,
                FormulaOperator.EXECUTION_EVENT,
                FormulaOperator.DEPENDENCY_ORDER,
                FormulaOperator.DEADLINE,
            )
            and self.operands
        ):
            raise ContractValidationError(
                "%s does not accept operands" % operator.value
            )
        if (
            operator
            in (
                FormulaOperator.LIVENESS,
                FormulaOperator.SAFETY,
                FormulaOperator.GOAL_SATISFACTION,
            )
            and self.upper_bound is None
        ):
            raise ContractValidationError(
                "%s requires a finite upper_bound" % operator.value
            )

    @property
    def formula_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        return {
            "vocabulary_version": LOGIC_VOCABULARY_VERSION,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "operator": self.operator,
            "predicate": self.predicate.value if self.predicate is not None else None,
            "terms": self.terms,
            "operands": self.operands,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Formula":
        _schema(payload, cls.SCHEMA)
        result = cls(
            operator=payload.get("operator", FormulaOperator.ATOM),
            predicate=payload.get("predicate"),
            terms=tuple(payload.get("terms") or ()),
            operands=tuple(payload.get("operands") or ()),
            lower_bound=payload.get("lower_bound"),
            upper_bound=payload.get("upper_bound"),
            profile_id=payload.get("profile_id", "supervisor-reviewed"),
            profile_version=payload.get("profile_version", LOGIC_VOCABULARY_VERSION),
        )
        claimed = payload.get("formula_id") or payload.get("content_id")
        if claimed and claimed != result.formula_id:
            raise ContractValidationError(
                "formula content identity does not match payload"
            )
        return result


def atom(predicate: ReviewedPredicate, *terms: LogicTerm) -> Formula:
    return Formula(
        operator=FormulaOperator.ATOM, predicate=predicate, terms=tuple(terms)
    )


def negate(formula: Formula) -> Formula:
    return Formula(operator=FormulaOperator.NOT, operands=(formula,))


def conjunction(*formulas: Formula) -> Formula:
    return Formula(operator=FormulaOperator.AND, operands=tuple(formulas))


def disjunction(*formulas: Formula) -> Formula:
    return Formula(operator=FormulaOperator.OR, operands=tuple(formulas))


def implies(antecedent: Formula, consequent: Formula) -> Formula:
    return Formula(operator=FormulaOperator.IMPLIES, operands=(antecedent, consequent))


class DCECVocabulary:
    """Builders for the reviewed DCEC planning subset."""

    profile_id = "supervisor-dcec"
    version = DCEC_VOCABULARY_VERSION
    operators = tuple(DCECOperator)

    @staticmethod
    def _modal(
        operator: FormulaOperator, actor_id: str, formula: Formula, time: int
    ) -> Formula:
        return Formula(
            operator=operator,
            terms=(constant(TermSort.ACTOR, actor_id), constant(TermSort.TIME, time)),
            operands=(formula,),
            profile_id=DCECVocabulary.profile_id,
            profile_version=DCECVocabulary.version,
        )

    @classmethod
    def belief(cls, actor_id: str, formula: Formula, time: int) -> Formula:
        return cls._modal(FormulaOperator.BELIEF, actor_id, formula, time)

    @classmethod
    def knowledge(cls, actor_id: str, formula: Formula, time: int) -> Formula:
        return cls._modal(FormulaOperator.KNOWLEDGE, actor_id, formula, time)

    @classmethod
    def intention(cls, actor_id: str, formula: Formula, time: int) -> Formula:
        return cls._modal(FormulaOperator.INTENTION, actor_id, formula, time)

    @classmethod
    def obligation(cls, actor_id: str, formula: Formula, deadline: int) -> Formula:
        return cls._modal(FormulaOperator.OBLIGATION, actor_id, formula, deadline)

    @classmethod
    def permission(cls, actor_id: str, formula: Formula, time: int) -> Formula:
        return cls._modal(FormulaOperator.PERMISSION, actor_id, formula, time)

    @classmethod
    def prohibition(cls, actor_id: str, formula: Formula, time: int) -> Formula:
        return cls._modal(FormulaOperator.PROHIBITION, actor_id, formula, time)

    @classmethod
    def delegation(
        cls, delegator_id: str, delegatee_id: str, task_id: str, time: int
    ) -> Formula:
        return Formula(
            operator=FormulaOperator.DELEGATION,
            terms=(
                constant(TermSort.ACTOR, delegator_id),
                constant(TermSort.ACTOR, delegatee_id),
                constant(TermSort.TASK, task_id),
                constant(TermSort.TIME, time),
            ),
            profile_id=cls.profile_id,
            profile_version=cls.version,
        )

    @classmethod
    def execution_event(
        cls, actor_id: str, task_id: str, event_id: str, time: int
    ) -> Formula:
        return Formula(
            operator=FormulaOperator.EXECUTION_EVENT,
            terms=(
                constant(TermSort.ACTOR, actor_id),
                constant(TermSort.TASK, task_id),
                constant(TermSort.EVENT, event_id),
                constant(TermSort.TIME, time),
            ),
            profile_id=cls.profile_id,
            profile_version=cls.version,
        )


DCEC = DCECVocabulary
DCEC_VOCABULARY = DCECVocabulary()
REVIEWED_DCEC_OPERATORS = tuple(DCECOperator)


class TDFOLVocabulary:
    """Builders for bounded temporal/dependency properties."""

    profile_id = "supervisor-tdfol"
    version = TDFOL_VOCABULARY_VERSION
    properties = tuple(TDFOLProperty)

    @classmethod
    def dependency_order(cls, predecessor_id: str, successor_id: str) -> Formula:
        return Formula(
            operator=FormulaOperator.DEPENDENCY_ORDER,
            terms=(
                constant(TermSort.TASK, predecessor_id),
                constant(TermSort.TASK, successor_id),
            ),
            profile_id=cls.profile_id,
            profile_version=cls.version,
        )

    @classmethod
    def deadline(cls, task_id: str, deadline: int) -> Formula:
        return Formula(
            operator=FormulaOperator.DEADLINE,
            terms=(
                constant(TermSort.TASK, task_id),
                constant(TermSort.TIME, deadline),
            ),
            upper_bound=deadline,
            profile_id=cls.profile_id,
            profile_version=cls.version,
        )

    @classmethod
    def liveness(
        cls, formula: Formula, upper_bound: int, lower_bound: int = 0
    ) -> Formula:
        return Formula(
            operator=FormulaOperator.LIVENESS,
            operands=(formula,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            profile_id=cls.profile_id,
            profile_version=cls.version,
        )

    @classmethod
    def safety(
        cls, formula: Formula, upper_bound: int, lower_bound: int = 0
    ) -> Formula:
        return Formula(
            operator=FormulaOperator.SAFETY,
            operands=(formula,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            profile_id=cls.profile_id,
            profile_version=cls.version,
        )

    @classmethod
    def goal_satisfaction(cls, goal_id: str, upper_bound: int) -> Formula:
        goal_atom = atom(
            ReviewedPredicate.GOAL_SATISFIED,
            constant(TermSort.GOAL, goal_id),
        )
        return Formula(
            operator=FormulaOperator.GOAL_SATISFACTION,
            operands=(goal_atom,),
            lower_bound=0,
            upper_bound=upper_bound,
            profile_id=cls.profile_id,
            profile_version=cls.version,
        )


TDFOL = TDFOLVocabulary
TDFOL_VOCABULARY = TDFOLVocabulary()
REVIEWED_PREDICATES = tuple(ReviewedPredicate)
REVIEWED_TDFOL_PROPERTIES = tuple(TDFOLProperty)


@dataclass(frozen=True)
class TraceFact(CanonicalContract):
    SCHEMA: ClassVar[str] = TRACE_FACT_SCHEMA

    predicate: ReviewedPredicate
    terms: Tuple[LogicTerm, ...]

    def __post_init__(self) -> None:
        formula = atom(self.predicate, *tuple(_term(item) for item in self.terms))
        object.__setattr__(self, "predicate", formula.predicate)
        object.__setattr__(self, "terms", formula.terms)

    @property
    def fact_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        return {
            "vocabulary_version": TDFOL_VOCABULARY_VERSION,
            "predicate": self.predicate,
            "terms": self.terms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceFact":
        _schema(payload, cls.SCHEMA)
        return cls(
            predicate=payload.get("predicate", ReviewedPredicate.SAFE_STATE),
            terms=tuple(payload.get("terms") or ()),
        )


@dataclass(frozen=True)
class TraceStep(CanonicalContract):
    SCHEMA: ClassVar[str] = TRACE_STEP_SCHEMA

    index: int
    facts: Tuple[TraceFact, ...] = ()
    evidence_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index", _positive(self.index, "index", allow_zero=True)
        )
        facts: List[TraceFact] = []
        for item in self.facts:
            facts.append(
                item if isinstance(item, TraceFact) else TraceFact.from_dict(item)
            )
        facts.sort(key=lambda item: item.fact_id)
        if len({item.fact_id for item in facts}) != len(facts):
            raise ContractValidationError("trace-step facts must be unique")
        object.__setattr__(self, "facts", tuple(facts))
        object.__setattr__(
            self, "evidence_ids", _string_tuple(self.evidence_ids, "evidence_ids")
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "vocabulary_version": TDFOL_VOCABULARY_VERSION,
            "index": self.index,
            "facts": self.facts,
            "evidence_ids": self.evidence_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceStep":
        _schema(payload, cls.SCHEMA)
        return cls(
            index=payload.get("index", -1),
            facts=tuple(payload.get("facts") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
        )


def _string_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise ContractValidationError("%s must be a sequence of strings" % field_name)
    return tuple(
        sorted({_text(item, field_name=field_name, required=True) for item in values})
    )


@dataclass(frozen=True)
class FiniteTrace(CanonicalContract):
    SCHEMA: ClassVar[str] = FINITE_TRACE_SCHEMA

    steps: Tuple[TraceStep, ...]
    bound: int
    source_plan_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bound", _positive(self.bound, "bound", allow_zero=True)
        )
        object.__setattr__(
            self,
            "source_plan_id",
            _text(self.source_plan_id, field_name="source_plan_id", required=True),
        )
        steps: List[TraceStep] = []
        for item in self.steps:
            steps.append(
                item if isinstance(item, TraceStep) else TraceStep.from_dict(item)
            )
        steps.sort(key=lambda item: item.index)
        if not steps or steps[0].index != 0:
            raise ContractValidationError("finite traces must start at step zero")
        if [item.index for item in steps] != list(range(len(steps))):
            raise ContractValidationError(
                "finite trace step indexes must be contiguous"
            )
        if steps[-1].index > self.bound:
            raise ContractValidationError("finite trace exceeds its bound")
        object.__setattr__(self, "steps", tuple(steps))

    @property
    def trace_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        return {
            "vocabulary_version": TDFOL_VOCABULARY_VERSION,
            "source_plan_id": self.source_plan_id,
            "bound": self.bound,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FiniteTrace":
        _schema(payload, cls.SCHEMA)
        return cls(
            source_plan_id=payload.get("source_plan_id", ""),
            bound=payload.get("bound", -1),
            steps=tuple(payload.get("steps") or ()),
        )


def _fact_at(formula: Formula, trace: FiniteTrace, index: int) -> bool:
    if formula.operator is not FormulaOperator.ATOM:
        return evaluate_formula(formula, trace, index=index)
    wanted = TraceFact(predicate=formula.predicate, terms=formula.terms).fact_id
    return any(item.fact_id == wanted for item in trace.steps[index].facts)


def evaluate_formula(formula: Formula, trace: FiniteTrace, *, index: int = 0) -> bool:
    """Evaluate the reviewed subset over an explicit finite trace."""

    if index < 0 or index >= len(trace.steps):
        raise ContractValidationError("evaluation index is outside the finite trace")
    operator = formula.operator
    if operator is FormulaOperator.ATOM:
        return _fact_at(formula, trace, index)
    if operator is FormulaOperator.NOT:
        return not evaluate_formula(formula.operands[0], trace, index=index)
    if operator is FormulaOperator.AND:
        return all(
            evaluate_formula(item, trace, index=index) for item in formula.operands
        )
    if operator is FormulaOperator.OR:
        return any(
            evaluate_formula(item, trace, index=index) for item in formula.operands
        )
    if operator is FormulaOperator.IMPLIES:
        return (
            not evaluate_formula(formula.operands[0], trace, index=index)
        ) or evaluate_formula(formula.operands[1], trace, index=index)
    if operator is FormulaOperator.IFF:
        return evaluate_formula(
            formula.operands[0], trace, index=index
        ) == evaluate_formula(formula.operands[1], trace, index=index)
    if operator is FormulaOperator.DEPENDENCY_ORDER:
        predecessor = str(formula.terms[0].value)
        successor = str(formula.terms[1].value)
        predecessor_fact = atom(
            ReviewedPredicate.TASK_COMPLETED, constant(TermSort.TASK, predecessor)
        )
        successor_fact = atom(
            ReviewedPredicate.TASK_STARTED, constant(TermSort.TASK, successor)
        )
        predecessor_steps = [
            step.index
            for step in trace.steps
            if _fact_at(predecessor_fact, trace, step.index)
        ]
        successor_steps = [
            step.index
            for step in trace.steps
            if _fact_at(successor_fact, trace, step.index)
        ]
        return not successor_steps or (
            bool(predecessor_steps) and min(predecessor_steps) <= min(successor_steps)
        )
    if operator is FormulaOperator.DEADLINE:
        task_id = str(formula.terms[0].value)
        deadline = int(formula.terms[1].value)
        completed = atom(
            ReviewedPredicate.TASK_COMPLETED, constant(TermSort.TASK, task_id)
        )
        return any(
            step.index <= deadline and _fact_at(completed, trace, step.index)
            for step in trace.steps
        )
    if operator in (FormulaOperator.LIVENESS, FormulaOperator.GOAL_SATISFACTION):
        lower = formula.lower_bound or 0
        upper = min(
            formula.upper_bound or trace.bound, trace.bound, len(trace.steps) - 1
        )
        return any(
            evaluate_formula(formula.operands[0], trace, index=step)
            for step in range(lower, upper + 1)
        )
    if operator is FormulaOperator.SAFETY:
        lower = formula.lower_bound or 0
        upper = min(
            formula.upper_bound or trace.bound, trace.bound, len(trace.steps) - 1
        )
        return all(
            evaluate_formula(formula.operands[0], trace, index=step)
            for step in range(lower, upper + 1)
        )
    raise ContractValidationError(
        "%s has no reference finite-trace semantics" % operator.value
    )


@dataclass(frozen=True)
class FrameProjectionConfig(CanonicalContract):
    SCHEMA: ClassVar[str] = FRAME_PROJECTION_CONFIG_SCHEMA

    max_worlds: int = 32
    max_hops: int = 2
    max_nodes: int = 128
    max_edges: int = 256

    def __post_init__(self) -> None:
        for name in ("max_worlds", "max_nodes", "max_edges"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self, "max_hops", _positive(self.max_hops, "max_hops", allow_zero=True)
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "projection_version": FRAME_LOGIC_PROJECTION_VERSION,
            "max_worlds": self.max_worlds,
            "max_hops": self.max_hops,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrameProjectionConfig":
        _schema(payload, cls.SCHEMA)
        return cls(
            max_worlds=payload.get("max_worlds", 32),
            max_hops=payload.get("max_hops", 2),
            max_nodes=payload.get("max_nodes", 128),
            max_edges=payload.get("max_edges", 256),
        )


@dataclass(frozen=True)
class World:
    world_id: str
    trace_index: int
    fact_ids: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "world_id", _text(self.world_id, field_name="world_id", required=True)
        )
        object.__setattr__(
            self,
            "trace_index",
            _positive(self.trace_index, "trace_index", allow_zero=True),
        )
        object.__setattr__(self, "fact_ids", _string_tuple(self.fact_ids, "fact_ids"))
        object.__setattr__(
            self, "evidence_ids", _string_tuple(self.evidence_ids, "evidence_ids")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_id": self.world_id,
            "trace_index": self.trace_index,
            "fact_ids": list(self.fact_ids),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class AccessibilityRelation:
    source_world_id: str
    target_world_id: str
    relation: str = "next"

    def __post_init__(self) -> None:
        for name in ("source_world_id", "target_world_id", "relation"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_world_id": self.source_world_id,
            "target_world_id": self.target_world_id,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    kind: str = "evidence"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_id", _text(self.node_id, field_name="node_id", required=True)
        )
        object.__setattr__(
            self, "kind", _text(self.kind, field_name="kind", required=True)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "kind": self.kind}


@dataclass(frozen=True)
class EvidenceEdge:
    source_id: str
    target_id: str
    relation: str

    def __post_init__(self) -> None:
        for name in ("source_id", "target_id", "relation"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class FrameLogicProjection(CanonicalContract):
    SCHEMA: ClassVar[str] = FRAME_LOGIC_PROJECTION_SCHEMA

    source_plan_id: str
    source_trace_id: str
    config: FrameProjectionConfig
    worlds: Tuple[World, ...]
    accessibility_relations: Tuple[AccessibilityRelation, ...]
    evidence_nodes: Tuple[EvidenceNode, ...]
    evidence_edges: Tuple[EvidenceEdge, ...]
    seed_ids: Tuple[str, ...] = ()
    truncated: bool = False
    omitted_counts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_plan_id",
            _text(self.source_plan_id, field_name="source_plan_id", required=True),
        )
        object.__setattr__(
            self,
            "source_trace_id",
            _text(self.source_trace_id, field_name="source_trace_id", required=True),
        )
        if isinstance(self.config, Mapping):
            object.__setattr__(
                self, "config", FrameProjectionConfig.from_dict(self.config)
            )
        elif not isinstance(self.config, FrameProjectionConfig):
            raise ContractValidationError("config must be a FrameProjectionConfig")
        worlds: List[World] = []
        for item in self.worlds:
            if isinstance(item, World):
                worlds.append(item)
            elif isinstance(item, Mapping):
                worlds.append(
                    World(
                        world_id=item.get("world_id", ""),
                        trace_index=item.get("trace_index", -1),
                        fact_ids=tuple(item.get("fact_ids") or ()),
                        evidence_ids=tuple(item.get("evidence_ids") or ()),
                    )
                )
            else:
                raise ContractValidationError("worlds must contain World values")
        object.__setattr__(self, "worlds", tuple(worlds))
        relations: List[AccessibilityRelation] = []
        for item in self.accessibility_relations:
            if isinstance(item, AccessibilityRelation):
                relations.append(item)
            elif isinstance(item, Mapping):
                relations.append(
                    AccessibilityRelation(
                        source_world_id=item.get("source_world_id", ""),
                        target_world_id=item.get("target_world_id", ""),
                        relation=item.get("relation", "next"),
                    )
                )
            else:
                raise ContractValidationError(
                    "accessibility_relations must contain AccessibilityRelation values"
                )
        object.__setattr__(self, "accessibility_relations", tuple(relations))
        nodes: List[EvidenceNode] = []
        for item in self.evidence_nodes:
            if isinstance(item, EvidenceNode):
                nodes.append(item)
            elif isinstance(item, Mapping):
                nodes.append(
                    EvidenceNode(
                        node_id=item.get("node_id", ""),
                        kind=item.get("kind", "evidence"),
                    )
                )
            else:
                raise ContractValidationError(
                    "evidence_nodes must contain EvidenceNode values"
                )
        object.__setattr__(self, "evidence_nodes", tuple(nodes))
        edges: List[EvidenceEdge] = []
        for item in self.evidence_edges:
            if isinstance(item, EvidenceEdge):
                edges.append(item)
            elif isinstance(item, Mapping):
                edges.append(
                    EvidenceEdge(
                        source_id=item.get("source_id", ""),
                        target_id=item.get("target_id", ""),
                        relation=item.get("relation", ""),
                    )
                )
            else:
                raise ContractValidationError(
                    "evidence_edges must contain EvidenceEdge values"
                )
        object.__setattr__(self, "evidence_edges", tuple(edges))
        object.__setattr__(self, "seed_ids", _string_tuple(self.seed_ids, "seed_ids"))
        if not isinstance(self.truncated, bool):
            raise ContractValidationError("truncated must be a boolean")
        object.__setattr__(
            self,
            "omitted_counts",
            _mapping(self.omitted_counts, field_name="omitted_counts"),
        )
        world_ids = {item.world_id for item in self.worlds}
        node_ids = {item.node_id for item in self.evidence_nodes}
        if len(world_ids) != len(self.worlds):
            raise ContractValidationError("frame world identifiers must be unique")
        if len(node_ids) != len(self.evidence_nodes):
            raise ContractValidationError("evidence node identifiers must be unique")
        if len(self.worlds) > self.config.max_worlds:
            raise ContractValidationError("frame projection exceeds max_worlds")
        if len(self.evidence_nodes) > self.config.max_nodes:
            raise ContractValidationError("frame projection exceeds max_nodes")
        if len(self.evidence_edges) > self.config.max_edges:
            raise ContractValidationError("frame projection exceeds max_edges")
        for relation in self.accessibility_relations:
            if (
                relation.source_world_id not in world_ids
                or relation.target_world_id not in world_ids
            ):
                raise ContractValidationError(
                    "accessibility relations must reference projected worlds"
                )
        for edge in self.evidence_edges:
            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                raise ContractValidationError(
                    "projected evidence edges must reference projected nodes"
                )
        if set(self.seed_ids) - node_ids:
            raise ContractValidationError(
                "frame projection seed ids must reference projected nodes"
            )
        for name, value in self.omitted_counts.items():
            _positive(value, "omitted_counts.%s" % name, allow_zero=True)

    @property
    def projection_id(self) -> str:
        return self.content_id

    @property
    def proves_code(self) -> bool:
        return False

    @property
    def code_assurance(self) -> AssuranceLevel:
        return AssuranceLevel.UNVERIFIED

    def _payload(self) -> Dict[str, Any]:
        return {
            "projection_version": FRAME_LOGIC_PROJECTION_VERSION,
            "source_plan_id": self.source_plan_id,
            "source_trace_id": self.source_trace_id,
            "config": self.config,
            "worlds": tuple(item.to_dict() for item in self.worlds),
            "accessibility_relations": tuple(
                item.to_dict() for item in self.accessibility_relations
            ),
            "evidence_nodes": tuple(item.to_dict() for item in self.evidence_nodes),
            "evidence_edges": tuple(item.to_dict() for item in self.evidence_edges),
            "seed_ids": self.seed_ids,
            "truncated": self.truncated,
            "omitted_counts": self.omitted_counts,
            "proves_code": False,
            "code_assurance": AssuranceLevel.UNVERIFIED,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrameLogicProjection":
        _schema(payload, cls.SCHEMA)
        if payload.get("proves_code") not in (None, False):
            raise ContractValidationError("frame projections can never prove code")
        if payload.get("code_assurance") not in (
            None,
            AssuranceLevel.UNVERIFIED,
            AssuranceLevel.UNVERIFIED.value,
        ):
            raise ContractValidationError(
                "frame projections can only have unverified code assurance"
            )
        result = cls(
            source_plan_id=payload.get("source_plan_id", ""),
            source_trace_id=payload.get("source_trace_id", ""),
            config=payload.get("config") or {},
            worlds=tuple(payload.get("worlds") or ()),
            accessibility_relations=tuple(payload.get("accessibility_relations") or ()),
            evidence_nodes=tuple(payload.get("evidence_nodes") or ()),
            evidence_edges=tuple(payload.get("evidence_edges") or ()),
            seed_ids=tuple(payload.get("seed_ids") or ()),
            truncated=payload.get("truncated", False),
            omitted_counts=payload.get("omitted_counts") or {},
        )
        claimed = payload.get("projection_id") or payload.get("content_id")
        if claimed and claimed != result.projection_id:
            raise ContractValidationError(
                "frame projection content identity does not match payload"
            )
        return result


def project_frame_logic(
    trace: FiniteTrace,
    *,
    evidence_nodes: Sequence[EvidenceNode] = (),
    evidence_edges: Sequence[EvidenceEdge] = (),
    seed_ids: Sequence[str] = (),
    config: Optional[FrameProjectionConfig] = None,
) -> FrameLogicProjection:
    """Derive bounded worlds, accessibility, and an evidence neighborhood."""

    selected_config = config or FrameProjectionConfig()
    selected_steps = trace.steps[: selected_config.max_worlds]
    worlds = tuple(
        World(
            world_id="%s:world:%d" % (trace.trace_id, step.index),
            trace_index=step.index,
            fact_ids=tuple(item.fact_id for item in step.facts),
            evidence_ids=step.evidence_ids,
        )
        for step in selected_steps
    )
    relations = tuple(
        AccessibilityRelation(worlds[index].world_id, worlds[index + 1].world_id)
        for index in range(max(0, len(worlds) - 1))
    )

    nodes_by_id = {item.node_id: item for item in evidence_nodes}
    normalized_edges = sorted(
        evidence_edges, key=lambda item: (item.source_id, item.target_id, item.relation)
    )
    seeds = _string_tuple(seed_ids, "seed_ids")
    if not seeds:
        seeds = tuple(sorted(nodes_by_id)[: selected_config.max_nodes])
    unknown = set(seeds) - set(nodes_by_id)
    if unknown:
        raise ContractValidationError(
            "frame projection has unknown seed ids: %s" % ", ".join(sorted(unknown))
        )
    adjacency: Dict[str, List[EvidenceEdge]] = {}
    for edge in normalized_edges:
        if edge.source_id not in nodes_by_id or edge.target_id not in nodes_by_id:
            raise ContractValidationError(
                "evidence edges must reference declared nodes"
            )
        adjacency.setdefault(edge.source_id, []).append(edge)
        adjacency.setdefault(edge.target_id, []).append(edge)

    distances = {seed: 0 for seed in seeds}
    queue = deque(seeds)
    selected_ids: List[str] = []
    selected_edge_keys: Set[Tuple[str, str, str]] = set()
    while queue and len(selected_ids) < selected_config.max_nodes:
        node_id = queue.popleft()
        if node_id in selected_ids:
            continue
        selected_ids.append(node_id)
        distance = distances[node_id]
        if distance >= selected_config.max_hops:
            continue
        for edge in adjacency.get(node_id, ()):
            neighbor = edge.target_id if edge.source_id == node_id else edge.source_id
            if neighbor not in distances:
                distances[neighbor] = distance + 1
                queue.append(neighbor)
            selected_edge_keys.add((edge.source_id, edge.target_id, edge.relation))

    selected_id_set = set(selected_ids)
    bounded_edges = [
        edge
        for edge in normalized_edges
        if edge.source_id in selected_id_set
        and edge.target_id in selected_id_set
        and (edge.source_id, edge.target_id, edge.relation) in selected_edge_keys
    ][: selected_config.max_edges]
    bounded_nodes = tuple(nodes_by_id[item] for item in selected_ids)
    omitted = {
        "worlds": max(0, len(trace.steps) - len(worlds)),
        "nodes": max(0, len(nodes_by_id) - len(bounded_nodes)),
        "edges": max(0, len(normalized_edges) - len(bounded_edges)),
    }
    return FrameLogicProjection(
        source_plan_id=trace.source_plan_id,
        source_trace_id=trace.trace_id,
        config=selected_config,
        worlds=worlds,
        accessibility_relations=relations,
        evidence_nodes=bounded_nodes,
        evidence_edges=tuple(bounded_edges),
        seed_ids=seeds,
        truncated=any(value > 0 for value in omitted.values()),
        omitted_counts=omitted,
    )


# Explicit compatibility spellings for downstream compiler and validator work.
# They remain aliases to the same closed AST and bounded reference functions.
DCECFormula = Formula
TDFOLFormula = Formula
FormalLogicFormula = Formula
FrameLogicWorld = World
build_frame_logic_projection = project_frame_logic
evaluate_tdfol_formula = evaluate_formula


__all__ = [
    "DCEC_VOCABULARY_VERSION",
    "FINITE_TRACE_SCHEMA",
    "FORMULA_SCHEMA",
    "FRAME_LOGIC_PROJECTION_SCHEMA",
    "FRAME_LOGIC_PROJECTION_VERSION",
    "LOGIC_TERM_SCHEMA",
    "LOGIC_VOCABULARY_VERSION",
    "TDFOL_VOCABULARY_VERSION",
    "AccessibilityRelation",
    "DCEC",
    "DCEC_VOCABULARY",
    "DCECOperator",
    "DCECFormula",
    "DCECVocabulary",
    "EvidenceEdge",
    "EvidenceNode",
    "FiniteTrace",
    "FormalLogicFormula",
    "Formula",
    "FormulaOperator",
    "FrameLogicProjection",
    "FrameLogicWorld",
    "FrameProjectionConfig",
    "LogicTerm",
    "LogicVocabularyValidationError",
    "ReviewedPredicate",
    "REVIEWED_PREDICATES",
    "REVIEWED_DCEC_OPERATORS",
    "REVIEWED_TDFOL_PROPERTIES",
    "Sort",
    "TDFOL",
    "TDFOL_VOCABULARY",
    "TDFOLProperty",
    "TDFOLFormula",
    "TDFOLVocabulary",
    "Term",
    "TermKind",
    "TermSort",
    "TraceFact",
    "TraceStep",
    "World",
    "atom",
    "build_frame_logic_projection",
    "conjunction",
    "constant",
    "disjunction",
    "evaluate_formula",
    "evaluate_tdfol_formula",
    "implies",
    "negate",
    "project_frame_logic",
    "variable",
]
